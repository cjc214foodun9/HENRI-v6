#!/bin/bash
# G-ACT4 on Blackwell. Pinned SHA. No fabricated latency.
set -x
cd /workspace || exit 9
git clone -q https://github.com/cjc214foodun9/HENRI-v6.git henri
cd henri || exit 9
git checkout -q c40ccbe6d811d4a9e5b1430a7031d47afb7ac783
git rev-parse HEAD > /workspace/pinned_sha.txt

# Hardware facts first. These are the preconditions that fail silently.
nvidia-smi -L > /workspace/hw_nvidia_smi_L.txt 2>&1
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm,clocks.current.sm,power.limit,temperature.gpu --format=csv > /workspace/hw_query.csv 2>&1
python - > /workspace/hw_torch.json 2>/workspace/hw_torch.err <<'PY'
import json, torch
r = {"torch": torch.__version__, "cuda": torch.version.cuda,
     "avail": bool(torch.cuda.is_available()), "n": torch.cuda.device_count()}
if r["avail"]:
    p = torch.cuda.get_device_properties(0)
    r.update({"name": p.name, "cc": f"{p.major}.{p.minor}",
              "sm": p.multi_processor_count,
              "l2_cache_mb": getattr(p, "L2_cache_size", None),
              "mem_gb": round(p.total_memory / 2**30, 2)})
print(json.dumps(r, indent=2))
PY

# The deliverable: the harness decides PASS/FAIL/BLOCKED on real hardware.
python scripts/run_gact4_latency.py --out receipts/gact4_latency.json > /workspace/gact4.stdout 2> /workspace/gact4.stderr
echo "rc=$?" > /workspace/gact4.status

cp receipts/gact4_latency.json /workspace/ 2>/dev/null
cp receipts/gact4_latency.json.sha256 /workspace/ 2>/dev/null
echo DONE > /workspace/onstart.done
