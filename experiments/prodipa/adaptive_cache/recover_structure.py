#!/usr/bin/env python3
"""Retry failed device-selection scorers, then resume the complete experiment.

The original controller is allowed to finish its already-running CPU searches.
Only FAILED ESMFold jobs are retried; completed work and cost attempts are kept.
"""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
import time

from common import HERE, OUT, PY, dump, job, read


def main():
    lock = open(OUT / ".recovery.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    failed = [read(p) for p in sorted((OUT / "jobs").glob("*.json"))
              if read(p).get("status") == "FAILED"]
    for record in failed:
        if not any(str(c).endswith("/run_esmfold_properties.py") for c in record["command"]):
            raise RuntimeError(f"Unexpected failed job needs inspection: {record['name']}")
    def mark(stage):
        dump(OUT / "RECOVERY_STATUS.json", dict(stage=stage, updated_at_unix=time.time(),
             jobs=[r["name"] for r in failed]))
        print(stage, flush=True)
    def retry(record):
        command = list(record["command"])
        if "--device" not in command:
            command += ["--device", "cuda:0"]
        job(record["name"], command, record["paradigm"], record["method"],
            record["phase"], record["node"], gpus=1, free_mib=28000)
    mark("retrying_failed_structure_jobs")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(retry, failed))
    mark("structure_repaired_waiting_for_original_cpu_jobs")
    # Existing controller exits after its pending scorers finish and surface
    # the old failed futures. Never interrupt expensive in-flight searches.
    with open(OUT / ".pipeline.lock", "a") as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX)
    mark("resuming_full_pipeline")
    os.execv(PY, [PY, str(HERE / "run.py")])


if __name__ == "__main__":
    main()
