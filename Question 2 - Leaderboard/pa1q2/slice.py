"""Run queued jobs in time-limited slices (for machines that stop background processes).

    python -m pa1q2.slice JOBS_FILE BUDGET_SECONDS WORKER WORKERS
Worker w handles jobs w, w+W, w+2W ... in order; each call resumes the first unfinished job
and stops between epochs before the budget runs out. Prints a one-line status.
"""
import shlex
import sys
import time

from .train import parser, run, run_tag
from pathlib import Path


def main():
    jobs_file, budget, worker, workers = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    lines = [l for l in Path(jobs_file).read_text().splitlines() if l.strip()]
    mine = lines[worker::workers]
    start = time.time()
    for line in mine:
        a = parser().parse_args(shlex.split(line))
        if (Path(a.out) / f"{run_tag(a)}.json").exists():
            continue
        left = budget - (time.time() - start)
        if left < 30:
            break
        done = run(a, budget=left)
        if not done:
            break
    finished = sum((Path(parser().parse_args(shlex.split(l)).out) /
                    f"{run_tag(parser().parse_args(shlex.split(l)))}.json").exists() for l in mine)
    print(f"worker {worker}: {finished}/{len(mine)} finished")


if __name__ == "__main__":
    main()
