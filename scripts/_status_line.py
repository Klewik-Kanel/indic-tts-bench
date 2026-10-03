#!/usr/bin/env python3
"""One formatted line for one run, from its last step record on stdin.

Split out of scripts/status.sh rather than embedded in it. The embedded
version needed nested shell quoting to get an f-string past `python3 -c`
inside a single-quoted block, which is the kind of thing that works until
somebody edits it.

Reads RID and MAX_STEPS from the environment and nothing else, so it stays a
pure formatter with no knowledge of where runs live.
"""

import json
import os
import sys


def main() -> int:
    rid = os.environ.get("RID", "?")
    mx = int(os.environ.get("MAX_STEPS", "100000"))
    raw = sys.stdin.readline()
    try:
        d = json.loads(raw)
    except Exception:                                         # noqa: BLE001
        print(f"  {rid:<5s} unreadable last line")
        return 0

    step = int(d.get("step") or 0)
    elapsed = float(d.get("elapsed_s") or 0.0)
    rate = step / elapsed if elapsed > 0 else 0.0
    eta_h = ((mx - step) / rate / 3600) if rate > 0 else float("inf")
    pct = 100.0 * step / mx if mx else 0.0
    loss = float(d.get("loss") or 0.0)

    # The headline loss is a sum, and for FastSpeech 2 the mel term is under
    # one per cent of it, so the three largest components are the part worth
    # watching. Absent on runs launched before component logging landed.
    extra = ""
    comp = d.get("components") or {}
    if comp:
        top = sorted(comp.items(), key=lambda kv: -abs(kv[1]))[:3]
        extra = "  " + " ".join(
            f"{k.replace('loss_', '')}={v:.3g}" for k, v in top)

    eta = "   n/a" if eta_h == float("inf") else f"{eta_h:5.2f} h"
    print(f"  {rid:<5s} {step:>6d}/{mx} {pct:5.1f}%  loss {loss:9.3f}  "
          f"{rate:5.2f} it/s  eta {eta}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
