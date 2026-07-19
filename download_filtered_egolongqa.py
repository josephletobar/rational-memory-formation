#!/usr/bin/env python3
"""Resumably download the non-procedural EgoLongQA subset to the external drive."""

from __future__ import annotations

import getpass
import json
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "facebook/wearable-ai"
ANNOTATIONS = Path("/Users/jleto/Downloads/wearable_ai_2026_egolongqa_val_700.jsonl")
TARGET = Path("/Volumes/Crucial X9/theory-of-mind")
MIN_FREE_BYTES = 5 * 2**30

EXCLUDED = {
    f"{stem}.mp4"
    for stem in """
059a88dee3d0bddc 08a129d4d3455f91 1cae852183dce97b 1e4a671fbd96fd4a 1f254ca918b0ded0 1f2fa2af1378221e 2012d6ccb98cc62f 2077ea2f35583ff6 248697710277f829 2e3bc21cf6685807 307004bf807802e2 3409f3004e2a8078 35c648e328355f23 3ec8a158983b80dd 419eed589c74bb83 4959be3d4b58cf56 4e1bbce234587113 55e3e91da83cc21a 566f73525ea7f571 58f804c4985d0131 5c1c35f6a889d91d 61876f06c149d4c8 7b4fb7f6da738501 7cb70948d87b16e8 807c58c65ef0c3a7 886d2a6075229794 91eb47c80caaa8bb 9613123a4d0212a0 97155f5bb5abd234 9b2db5bc837eb13e 9f00cea9b9eef2d2 a2756d0d739a9644 a4cc59d31c1ff9f5 a510fbc0d1fdac03 a79da05071fe6c1e bc6cc236f240d2cd c6866ef535b21a76 dcdb8d8cd7889358 e93c3bfaeed14fe6 ecf8e9e9c8fcbd8a f267337fae95a881 f3535a6623c8d73c fc12d4e78cb8b907
056734786b556b0f 06615b19ef5373ca 0df064a955d50897 0efc268990f5588b 205a1ef3950ef283 31307eb6d9e335b3 31a5289ea39eca44 34e6a4629781783d 4ad1f0bdbc6f74dc 52a951fc79d7bdd0 5a96fde1c24cb409 5bcb19219734d71f 614db0132c094efa 629ab8ccd0a3f554 633c3483dd64f88c 6487bd211ab765aa 6c34f96dae9476ce 6dcb7f9fc04a3283 7593bd8121dcfd39 7a5c4239bc6b4a33 8a3a19f9c55cf5a9 93d3d7020fcaff98 985053519dfa50d5 a035a49d70ac199e a12fcce1fd9303be bde055e6fe9d2c08 dcc8378bca4fb4d8 df020e5f1eb92edd e0350daea52c7ddf e58aa85ab695d9cf e7fcdfd76103e193 e9f21972ff8c3870 ebd8e191c386b280 fe19b9dc6e384e4a
1a89eaa35120072a 1b0287a0da8011d5 1cc66eed89b44527 239c596fbc5ff872 280f3781277b0b97 295e35c4ad8632a4 2c50721b06e0bdb2 2ec4cbd4fdf4a1b3 2efdb9f426131303 2fc07122287502b0 3aa7a20249708db0 3b98ab8aa7763741 469f55fef5c7c288 4a96bbba4873d9a8 51b376fd7498147e 5910dd06ac368119 59a2f8377ce14120 5e1ba77a7f73c84d 67d2bd73430ff38e 80150ed4385f86a9 8421a42d1b5d314a 871fb7842cf193f1 88e8cf11f150ba5e 93026d95399f1fd3 9b7588c59c197d64 b3fba3c9b747e519 bb165f4c81827b82 bc1c13c75646fb2f cd3527f83b3036e2 dbafb0e35a78a21d e3f2d04be59af3eb e4947cad50181471 f4d2a0877e75856b
14a8ff92adb183c7 1ac045ca12edbb6a 1ae2ae78bd7cf791 32d790a85a4b9997 33ebb9c842c4e4d9 3a28a265e1f29959 3be840b7358c22ee 451fb61d25ac9294 461825ec4f8e89d4 4ceceb8b2234316b 58dcdf7f4b6b4878 5ac7b1152960f1e4 620e719ae9975498 62b5891eb1a85b09 741fcd43a96bd40e 81136397be32b026 888ab584bea8ddee 88c14ad96aa2ed09 8b1c280ae366ad2d 8f6a3caa4cf18e28 91a120b16bcf6c52 92c727a3edff6092 9da6c6890a9414bb ac411836489d5df4 b32b516fae8b102c b67686a226da2f5a c060b9a8c96fcc0f c663d84f69482b36 d4ad94f700885cee d938c527010d0246 dc93aa20988701ba
08910129d9d6ff34 166ed6f9ba6e5dea 1881e98bd8759b6f 22d2c11a300d339d 3cfc9122829c55aa 44107e68e219c362 59160e74f6de369c 719245d9778f41a8 76cd872d8785dae8 7ac4e2e4bc2ecc3d 8489ba53a98bcdbe a65dec1048bd5e15 aba51830e340f122 ba88525691af500d c7403ae0345a15f9 c8cb286a8a6583c7 cfd872ca200e9529 e1cf4266007a09fc ef75ac3d92198596
1a96d31d14d53569 5ab7030a02f30518 674a68e938dbd25b
68f002c1fab08f95 741fcd43a96bd40e 77d6d602f07f3bc8 88c14ad96aa2ed09 8b1c280ae366ad2d b291a5566d2fb7a6 b32b516fae8b102c e40865d30d1c1594 f06b20e6dc00da11
""".split()
}


def main() -> None:
    if len(EXCLUDED) != 168:
        raise RuntimeError(f"Expected 168 exclusions, found {len(EXCLUDED)}")
    rows = [json.loads(line) for line in ANNOTATIONS.open()]
    wanted = [row["video_path"] for row in rows if row["video_path"] not in EXCLUDED]
    if len(wanted) != 532:
        raise RuntimeError(f"Expected 532 retained videos, found {len(wanted)}")

    token = getpass.getpass("Hugging Face token: ")
    api = HfApi(token=token)
    identity = api.whoami()
    print(f"Authenticated as {identity.get('name', 'unknown')}")
    tree = list(
        api.list_repo_tree(
            REPO_ID,
            path_in_repo="egolongqa/val",
            repo_type="dataset",
            expand=True,
        )
    )
    sizes = {Path(item.path).name: item.size for item in tree}
    if len(sizes) != 700:
        raise RuntimeError(f"Expected metadata for 700 videos, found {len(sizes)}")

    TARGET.mkdir(parents=True, exist_ok=True)
    pending = []
    for name in wanted:
        destination = TARGET / name
        expected = sizes[name]
        if destination.exists() and destination.stat().st_size == expected:
            continue
        if destination.exists():
            backup = destination.with_name(f"{destination.name}.invalid-{int(time.time())}")
            destination.replace(backup)
            print(f"Preserved size-mismatched file as {backup.name}")
        pending.append(name)

    pending_bytes = sum(sizes[name] for name in pending)
    free_bytes = shutil.disk_usage(TARGET).free
    print(
        f"Retained: {len(wanted)}; already valid: {len(wanted) - len(pending)}; "
        f"pending: {len(pending)} ({pending_bytes / 2**30:.2f} GiB); "
        f"free: {free_bytes / 2**30:.2f} GiB"
    )
    if free_bytes - pending_bytes < MIN_FREE_BYTES:
        raise RuntimeError(
            "Insufficient safety buffer: download would leave less than 5 GiB free"
        )

    failures: list[str] = []
    for index, name in enumerate(pending, 1):
        print(f"[{index}/{len(pending)}] downloading {name}", flush=True)
        filename = f"egolongqa/val/{name}"
        for attempt in range(1, 4):
            try:
                downloaded = Path(
                    hf_hub_download(
                        REPO_ID,
                        filename=filename,
                        repo_type="dataset",
                        token=token,
                        local_dir=TARGET,
                    )
                )
                if downloaded.stat().st_size != sizes[name]:
                    raise IOError(
                        f"size mismatch: {downloaded.stat().st_size} != {sizes[name]}"
                    )
                downloaded.replace(TARGET / name)
                break
            except Exception as exc:
                print(f"  attempt {attempt}/3 failed: {exc}", flush=True)
                if attempt == 3:
                    failures.append(name)
                else:
                    time.sleep(5 * attempt)
        if name not in failures:
            left = sum(sizes[n] for n in pending[index:])
            print(f"  saved; approximately {left / 2**30:.2f} GiB remaining", flush=True)

    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        raise SystemExit(1)
    print("DOWNLOAD COMPLETE: all 532 retained videos validated")


if __name__ == "__main__":
    main()
