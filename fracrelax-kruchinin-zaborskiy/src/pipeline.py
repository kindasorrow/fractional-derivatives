r"""Run the whole pipeline and write a fresh `results/pipeline_log.txt`.

    python -m src.pipeline                 # everything
    python -m src.pipeline --skip train    # re-use existing checkpoints
    python -m src.pipeline --only validate datagen

Each stage is the same `python -m src.<stage>` command the README lists, run in
a child process (so the torch-free stages keep their torch-free process). The log
is overwritten on every run and records the date, versions, the exact command,
return code and wall time of each stage, followed by that stage's output, so
it can never describe an older run than the results next to it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import subprocess
import sys
import time

from .config import RESULTS

import os

STAGES = [
    ("selftest", ["-m", "src.selftest"]),
    ("validate", ["-m", "src.validate"]),
    ("datagen", ["-m", "src.datagen"]),
    ("train", ["-m", "src.train", "--all"]),
    ("bench_classical", ["-m", "src.bench_classical"]),
    ("bench", ["-m", "src.bench"]),
    ("analyze", ["-m", "src.analyze"]),
    ("analyze_kernels", ["-m", "src.analyze_kernels"]),
    ("figures", ["-m", "src.figures"]),
    ("report", ["-m", "src.report"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    todo = [(n, c) for n, c in STAGES
            if (args.only is None or n in args.only) and n not in args.skip]

    path = os.path.join(RESULTS, "pipeline_log.txt")
    with open(path, "w", encoding="utf-8") as log:
        def w(s=""):
            log.write(s + "\n")
            log.flush()
            print(s)

        w(f"# pipeline run {dt.datetime.now().isoformat(timespec='seconds')}")
        w(f"# python {platform.python_version()} on {platform.platform()}")
        try:
            import torch
            w(f"# torch {torch.__version__}, cuda={torch.cuda.is_available()}")
        except Exception as e:                      # torch-free machine
            w(f"# torch unavailable ({e})")
        for name, cmd in todo:
            w(f"\n===== {name}: python {' '.join(cmd)}")
            t0 = time.time()
            p = subprocess.Popen([sys.executable] + cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace")
            for line in p.stdout:
                log.write(line)
            rc = p.wait()
            w(f"===== {name}: exit {rc}, {time.time() - t0:.0f} s")
            if rc != 0:
                w(f"# STOPPED: stage '{name}' failed; later stages not run")
                sys.exit(rc)
        w("\n# all requested stages finished")


if __name__ == "__main__":
    main()
