#!/usr/bin/env python3
"""SLURM utilities for LongQA generation.

Generates a split-and-merge script for one LongQA dataset shard per node.
Each node runs the same generation script on an input shard and writes a
`pred_shard_$SLURM_PROCID.jsonl`. A final merge pass interleaves shards
into the requested output.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys


def _split_jsonl(input_path: str, num_shards: int, output_dir: str) -> list[str]:
    with open(input_path) as f:
        lines = [line for line in f if line.strip()]

    shard_paths = []
    for i in range(num_shards):
        shard_lines = lines[i::num_shards]
        shard_path = os.path.join(output_dir, f"shard_{i}.jsonl")
        with open(shard_path, "w") as f:
            f.writelines(shard_lines)
        shard_paths.append(shard_path)
    return shard_paths


def _merge_shards(shard_dir: str, output_path: str, num_shards: int) -> int:
    shard_data: dict[int, list[str]] = {}
    missing: list[int] = []
    total = 0
    for i in range(num_shards):
        shard_path = os.path.join(shard_dir, f"pred_shard_{i}.jsonl")
        try:
            with open(shard_path) as f:
                shard_data[i] = [line for line in f if line.strip()]
        except FileNotFoundError:
            missing.append(i)
            shard_data[i] = []
        total += len(shard_data[i])

    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{num_shards} shard(s) missing: "
            + ", ".join(str(m) for m in missing)
        )

    max_per_shard = max(len(v) for v in shard_data.values()) if shard_data else 0
    with open(output_path, "w") as f:
        for idx in range(max_per_shard):
            for shard_id in range(num_shards):
                if idx < len(shard_data[shard_id]):
                    f.write(shard_data[shard_id][idx])
                    if not shard_data[shard_id][idx].endswith("\n"):
                        f.write("\n")
    return total


def _validate_sbatch_params(partition: str, reservation: str, time_limit: str) -> None:
    _sbatch_safe = re.compile(r"^[A-Za-z0-9_:.,/-]+$")
    for name, value in [
        ("partition", partition),
        ("reservation", reservation),
        ("time_limit", time_limit),
    ]:
        if value and not _sbatch_safe.match(value):
            raise ValueError(
                f"Invalid {name}={value!r}: must contain only "
                "alphanumeric characters, underscores, colons, dots, "
                "commas, hyphens, and forward slashes."
            )


def _resolve_conda_env(conda_env: str, conda_base: str) -> tuple[str, str, str, str]:
    if not conda_env:
        return "", "", "", ""

    conda_sh = ""
    if conda_base:
        conda_sh = os.path.join(conda_base, "etc", "profile.d", "conda.sh")
        if not os.path.exists(conda_sh):
            raise ValueError(f"conda.sh not found at {conda_sh}")
    else:
        env_parent = os.path.dirname(conda_env)
        if os.path.basename(env_parent) == "envs":
            inferred_base = os.path.dirname(env_parent)
        else:
            inferred_base = conda_env
        candidate = os.path.join(inferred_base, "etc", "profile.d", "conda.sh")
        if os.path.exists(candidate):
            conda_sh = candidate

    q_conda_sh = shlex.quote(conda_sh) if conda_sh else ""
    q_conda = shlex.quote(conda_env)
    conda_source = f"source {q_conda_sh}\n" if conda_sh else ""
    env_setup = f"{conda_source}conda activate {q_conda}\n"

    conda_exports = f"export _CONDA_ENV={q_conda}\n"
    if conda_sh:
        conda_exports += f"export _CONDA_SH={q_conda_sh}\n"
        srun_env_setup = 'source "${_CONDA_SH}" && conda activate "${_CONDA_ENV}"'
    else:
        srun_env_setup = 'conda activate "${_CONDA_ENV}"'

    return env_setup, conda_exports, srun_env_setup, q_conda


def _build_extra_sbatch(partition: str, reservation: str) -> str:
    sbatch_lines = []
    if partition:
        sbatch_lines.append(f"#SBATCH --partition={partition}")
    if reservation:
        sbatch_lines.append(f"#SBATCH --reservation={reservation}")
    return "\n".join(sbatch_lines)


def add_slurm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--slurm-nodes", type=int, default=0, help="Number of SLURM nodes for generation.")
    parser.add_argument(
        "--slurm-partition",
        type=str,
        default="",
        help="SLURM partition to submit to.",
    )
    parser.add_argument(
        "--slurm-reservation",
        type=str,
        default="",
        help="SLURM reservation (optional).",
    )
    parser.add_argument(
        "--slurm-time",
        type=str,
        default="4:00:00",
        help="Job time limit (e.g., 4:00:00).",
    )
    parser.add_argument(
        "--slurm-gpus",
        type=int,
        default=8,
        help="GPUs per node.",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="",
        help="Optional conda env path to activate inside sbatch.",
    )
    parser.add_argument(
        "--conda-base",
        type=str,
        default="",
        help="Optional conda base path (for conda.sh lookup).",
    )


def submit(
    script: str,
    input_path: str,
    output_path: str,
    num_nodes: int,
    extra_args: list[str],
    partition: str = "",
    reservation: str = "",
    conda_env: str = "",
    conda_base: str = "",
    gpus_per_node: int = 8,
    time_limit: str = "4:00:00",
    post_merge_commands: list[str] | None = None,
) -> str:
    _validate_sbatch_params(partition, reservation, time_limit)

    work_dir = os.path.dirname(os.path.abspath(script))
    _sbatch_unsafe = re.compile(r"[\s$`\"'\\!;|&<>(){}%]")
    unsafe_match = _sbatch_unsafe.search(work_dir)
    if unsafe_match:
        raise ValueError(
            f"work_dir contains unsafe character {unsafe_match.group()!r} "
            f"(SBATCH directives cannot quote paths): {work_dir}"
        )
    os.makedirs(os.path.join(work_dir, "logs"), exist_ok=True)

    output_stem = re.sub(
        r"[^A-Za-z0-9_.-]",
        "_",
        os.path.splitext(os.path.basename(output_path))[0],
    )
    shard_dir = os.path.join(
        os.path.dirname(os.path.abspath(output_path)), f"_shards_{output_stem}"
    )
    os.makedirs(shard_dir, exist_ok=True)

    print(f"Splitting {input_path} into {num_nodes} shards...")
    shard_inputs = _split_jsonl(input_path, num_nodes, shard_dir)
    for i, p in enumerate(shard_inputs):
        with open(p) as f:
            n = sum(1 for _ in f)
        print(f"  Shard {i}: {n} samples")

    extra_sbatch = _build_extra_sbatch(partition, reservation)
    env_setup, conda_exports, srun_env_setup, _ = _resolve_conda_env(
        conda_env, conda_base
    )

    extra_str = " ".join(shlex.quote(a) for a in extra_args)
    q_extra_str = shlex.quote(extra_str) if extra_str else "''"
    q_work_dir = shlex.quote(work_dir)
    q_script = shlex.quote(script)
    q_shard_dir = shlex.quote(shard_dir)
    q_output_path = shlex.quote(output_path)
    post_merge = "\n".join(post_merge_commands) if post_merge_commands else ""

    sbatch_script = f"""#!/bin/bash
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks={num_nodes}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time={time_limit}
#SBATCH --job-name=longqa_eval
#SBATCH --output={work_dir}/logs/{output_stem}_%j.log
#SBATCH --error={work_dir}/logs/{output_stem}_%j.log
{extra_sbatch}
set -euo pipefail
export PYTHONUNBUFFERED=1
cd {q_work_dir}
{env_setup}
{conda_exports}export _SHARD_DIR={q_shard_dir}
export _SCRIPT={q_script}
export _OUTPUT_PATH={q_output_path}
export _EXTRA_ARGS={q_extra_str}
srun bash -c '
  {srun_env_setup}
  SHARD_IN="${{_SHARD_DIR}}/shard_${{SLURM_PROCID}}.jsonl"
  SHARD_OUT="${{_SHARD_DIR}}/pred_shard_${{SLURM_PROCID}}.jsonl"
  echo "Node $(hostname): shard ${{SLURM_PROCID}}/{num_nodes}"
  eval python3 "${{_SCRIPT}}" --input "$SHARD_IN" --output "$SHARD_OUT" --no-eval ${{_EXTRA_ARGS}}
'

echo "All {num_nodes} shards done. Merging..."
python3 -c "
import os
from slurm_runner import _merge_shards
total = _merge_shards(os.environ['_SHARD_DIR'], os.environ['_OUTPUT_PATH'], {num_nodes})
print(f'Merged {{total}} predictions into ' + os.environ['_OUTPUT_PATH'])
"
{post_merge}
echo "DONE"
"""
    script_path = os.path.join(work_dir, f"_slurm_{output_stem}.sh")
    with open(script_path, "w") as f:
        f.write(sbatch_script)

    result = subprocess.run(
        ["sbatch", "--parsable", script_path],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}", file=sys.stderr)
        raise RuntimeError("sbatch failed")

    job_id = result.stdout.strip()
    print(f"\nSubmitted SLURM {job_id} ({num_nodes} nodes)")
    print(f"Monitor: sacct -j {job_id} --format=JobID,State,Elapsed")
    print(f"Logs: {work_dir}/logs/{output_stem}_{job_id}.log")

    return job_id
