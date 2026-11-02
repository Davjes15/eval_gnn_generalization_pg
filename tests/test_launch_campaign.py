"""Tests for the stdlib campaign launcher."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import launch_campaign


FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def _args(tmp, *extra):
    return [
        "--data_a", os.path.join(tmp, "data_a"),
        "--data_b", os.path.join(tmp, "data_b"),
        "--out", os.path.join(tmp, "out"),
        "--ckpt", os.path.join(tmp, "ckpt"),
        "--logs", os.path.join(tmp, "logs"),
        *extra,
    ]


def test_default_job_plan():
    print("\nDefault campaign plan")
    with tempfile.TemporaryDirectory() as tmp:
        args = launch_campaign._parser().parse_args(_args(tmp))
        jobs = launch_campaign.build_jobs(args)
    counts = {
        arm: sum(job["arm"] == arm for job in jobs)
        for arm in ("within", "cross", "ood")
    }
    job_ids = [job["job_id"] for job in jobs]
    out_dirs = [job["out_dir"] for job in jobs]
    check("default total is 168 jobs", len(jobs) == 168, str(len(jobs)))
    check("per-arm counts are 28/28/112",
          counts == {"within": 28, "cross": 28, "ood": 112}, str(counts))
    check("nnconv is first in the longest-first order",
          jobs[0]["job_id"].startswith("within_nnconv_")
          or jobs[0]["job_id"].startswith("cross_nnconv_"),
          jobs[0]["job_id"])
    check("job IDs are unique", len(job_ids) == len(set(job_ids)))
    check("output directories are unique", len(out_dirs) == len(set(out_dirs)))
    ood_jobs = [job for job in jobs if job["arm"] == "ood"]
    check("OOD commands specify held-out folds and batch size 96",
          all("--held_out" in job["cmd"]
              and job["cmd"][job["cmd"].index("--held_out") + 1]
              in launch_campaign.FOLDS
              and job["cmd"][job["cmd"].index("--batch_size_ood") + 1] == "96"
              for job in ood_jobs))
    within_jobs = [job for job in jobs if job["arm"] == "within"]
    check("within commands use data_a",
          all(job["cmd"][job["cmd"].index("--data_dir") + 1]
              == args.data_a for job in within_jobs))


def _fake_job(job_id, arm, command, out_dir):
    return {
        "job_id": job_id,
        "arm": arm,
        "cmd": command,
        "out_dir": out_dir,
        "cost": 1.0,
    }


def test_run_pool_retries_markers_and_ledger():
    print("\nPool retries, output markers, and append-only ledger")
    with tempfile.TemporaryDirectory() as tmp:
        out_root = os.path.join(tmp, "out")
        logs_dir = os.path.join(tmp, "logs")
        ledger_path = os.path.join(out_root, "campaign_ledger.jsonl")
        success_dir = os.path.join(out_root, "within_success")
        fail_dir = os.path.join(out_root, "cross_always_fail")
        retry_dir = os.path.join(out_root, "ood_retry_once")
        counter_path = os.path.join(tmp, "attempt_count")
        write_outputs = (
            "from pathlib import Path; import json,sys; "
            "p=Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=True); "
            "(p/'summary.json').write_text('{}'); "
            "(p/'within_grid.csv').write_text('ok\\n')"
        )
        success = _fake_job(
            "within_success", "within",
            [sys.executable, "-c", write_outputs, success_dir], success_dir,
        )
        always_fail = _fake_job(
            "cross_always_fail", "cross",
            [sys.executable, "-c", "import sys; sys.exit(4)"], fail_dir,
        )
        retry_code = (
            "from pathlib import Path\n"
            "import sys\n"
            "counter = Path(sys.argv[1])\n"
            "out_dir = Path(sys.argv[2])\n"
            "count = int(counter.read_text()) if counter.exists() else 0\n"
            "counter.write_text(str(count + 1))\n"
            "if count == 0:\n"
            "    sys.exit(5)\n"
            "out_dir.mkdir(parents=True, exist_ok=True)\n"
            "(out_dir / 'summary.json').write_text('{}')\n"
            "(out_dir / 'ood.csv').write_text('ok\\n')\n"
        )
        retry = _fake_job(
            "ood_retry_once", "ood",
            [sys.executable, "-c", retry_code, counter_path, retry_dir],
            retry_dir,
        )

        failed = launch_campaign.run_pool(
            [success, always_fail, retry], 2, 1, logs_dir, ledger_path
        )
        check("only the exhausted job is reported failed",
              failed == ["cross_always_fail"], str(failed))
        check("successful output writes DONE marker",
              os.path.isfile(os.path.join(success_dir, "DONE.json")))
        check("always-failing job writes FAILED, not DONE",
              os.path.isfile(os.path.join(fail_dir, "FAILED.json"))
              and not os.path.exists(os.path.join(fail_dir, "DONE.json")))
        retry_marker = os.path.join(retry_dir, "DONE.json")
        check("retry succeeds on its second attempt",
              os.path.isfile(retry_marker)
              and json.load(open(retry_marker))["attempts"] == 2)

        with open(ledger_path) as fh:
            ledger = [json.loads(line) for line in fh]
        check("ledger contains starts, finishes, and failures",
              {entry["event"] for entry in ledger} == {"start", "finish", "fail"})
        check("retry has one failure and one finish",
              sum(entry["event"] == "fail"
                  and entry["job_id"] == "ood_retry_once" for entry in ledger) == 1
              and sum(entry["event"] == "finish"
                      and entry["job_id"] == "ood_retry_once"
                      for entry in ledger) == 1)

        ledger_before = len(ledger)
        success_log = os.path.join(logs_dir, "within_success.log")
        success_log_size = os.path.getsize(success_log)
        second_run = launch_campaign.run_pool(
            [success, retry], 2, 1, logs_dir, ledger_path
        )
        with open(ledger_path) as fh:
            ledger_after = [json.loads(line) for line in fh]
        check("relaunch skips jobs with DONE markers",
              not second_run and len(ledger_after) == ledger_before
              and os.path.getsize(success_log) == success_log_size)


def test_dry_run_and_status():
    print("\nDry-run and status modes do not launch work")
    with tempfile.TemporaryDirectory() as tmp:
        dry_args = _args(tmp, "--dry_run", "--models", "gcn",
                         "--seeds", "0", "--seeds_nnconv", "0")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = launch_campaign.main(dry_args)
        check("dry-run succeeds and lists one job per arm",
              result == 0 and "jobs=6" in output.getvalue(),
              output.getvalue().splitlines()[-1])
        check("dry-run creates no output, checkpoint, or log directories",
              not any(os.path.exists(os.path.join(tmp, name))
                      for name in ("out", "ckpt", "logs")))

        status_args = _args(
            tmp, "--status", "--arms", "within", "cross", "ood",
            "--models", "gcn", "--seeds", "0", "--seeds_nnconv", "0",
        )
        jobs = launch_campaign.build_jobs(
            launch_campaign._parser().parse_args(status_args)
        )
        done_job = next(job for job in jobs if job["arm"] == "within")
        failed_job = next(job for job in jobs if job["arm"] == "cross")
        os.makedirs(done_job["out_dir"])
        os.makedirs(failed_job["out_dir"])
        with open(os.path.join(done_job["out_dir"], "DONE.json"), "w") as fh:
            json.dump({"job_id": done_job["job_id"]}, fh)
        with open(os.path.join(failed_job["out_dir"], "FAILED.json"), "w") as fh:
            json.dump({"job_id": failed_job["job_id"]}, fh)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = launch_campaign.main(status_args)
        lines = output.getvalue().splitlines()
        check("status succeeds without running jobs", result == 0)
        check("status counts done, failed, and pending per arm",
              lines == [
                  "within: done=1 failed=0 pending=0",
                  "cross: done=0 failed=1 pending=0",
                  "ood: done=0 failed=0 pending=4",
              ], str(lines))
        ledger_path = os.path.join(tmp, "out", "campaign_ledger.jsonl")
        check("status creates no logs or ledger",
              not os.path.exists(os.path.join(tmp, "logs"))
              and not os.path.exists(ledger_path))


def main():
    test_default_job_plan()
    test_run_pool_retries_markers_and_ledger()
    test_dry_run_and_status()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S):")
        for failure in FAILURES:
            print(f" - {failure}")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
