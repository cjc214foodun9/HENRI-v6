import json, sys, time
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import run_holarch_ac1_homodyne_egress as ac1
from pathlib import Path

started = time.perf_counter()
payload = ac1.measure(
    ac1.RECEIPTS,
    guard_tokens=8192,
    dump_per_token=str(ac1.RECEIPTS / "_work_ae" / "ac1_per_token.json"),
)
print(json.dumps({
    "sub_bpc": payload["subslice"]["bits_per_char"],
    "sub_top1": payload["subslice"]["top1"],
    "full_bpc": payload["full_split"]["bits_per_char"],
    "full_top1": payload["full_split"]["top1"],
    "seconds": round(time.perf_counter() - started, 1),
}))
