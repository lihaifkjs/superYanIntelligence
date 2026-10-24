"""云机执行：分片下载 Qwen3.6-35B-A3B，散布到 /dev/shm + autodl-tmp，建统一 symlink 视图。"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from modelscope.hub.api import HubApi

MODEL = "Qwen/Qwen3.6-35B-A3B"
SHM_DIR = Path("/dev/shm/q36_shards")
TMP_DIR = Path("/root/autodl-tmp/q36_shards")
UNIFIED = Path("/root/autodl-tmp/q36")
SHM_BUDGET = 55e9
WORKERS = 3
MS = "/root/miniconda3/bin/modelscope"


def dl(path: str, local_dir: Path):
    out = local_dir / path
    expected = FILESIZES[path]
    if out.exists() and out.stat().st_size == expected:
        print(f"skip {path}", flush=True)
        return
    subprocess.run(
        [MS, "download", "--model", MODEL, path, "--local_dir", str(local_dir)],
        check=True,
    )


api = HubApi()
files = api.get_model_files(MODEL, recursive=True)
FILESIZES = {f["Path"]: f["Size"] for f in files if f.get("Type") != "tree"}
shards = sorted(p for p in FILESIZES if p.endswith(".safetensors"))
small = [p for p in FILESIZES if not p.endswith(".safetensors")]

SHM_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)
UNIFIED.mkdir(exist_ok=True)

plan = []
shm_used = 0.0
for p in shards:
    s = FILESIZES[p]
    if shm_used + s <= SHM_BUDGET:
        plan.append((p, SHM_DIR))
        shm_used += s
    else:
        plan.append((p, TMP_DIR))
tmp_used = sum(FILESIZES[p] for p, d in plan if d == TMP_DIR)
print(f"plan: {len(plan)} shards | shm {shm_used/1e9:.1f}GB | tmp {tmp_used/1e9:.1f}GB", flush=True)

jobs = [(p, d) for p, d in plan] + [(p, TMP_DIR) for p in small]
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    list(ex.map(lambda a: dl(*a), jobs))

for p, d in plan + [(p, TMP_DIR) for p in small]:
    src = d / p
    dst = UNIFIED / p
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.symlink(src, dst)

print("DONE unified:", UNIFIED, flush=True)
