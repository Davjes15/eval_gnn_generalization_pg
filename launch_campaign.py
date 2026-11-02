#!/usr/bin/env python3
"""Launch interrupt-resilient campaign shards with a bounded process pool."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import signal
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = ("arma_gnn", "nnconv", "transformer", "gin", "gat", "gcn")
SEEDS = (0, 100, 300, 700, 1000)
SEEDS_NNCONV = (0, 100, 300)
FOLDS = ("UK", "IEEE118", "IEEE39", "IEEE24")
MODEL_SECONDS = {
    "nnconv": 64.0,
    "arma_gnn": 15.0,
    "gin": 1.7,
    "gat": 1.6,
    "transformer": 1.5,
    "gcn": 0.9,
}
OOD_LOAD = {"IEEE24": 0.2, "IEEE39": 0.25, "UK": 0.53, "IEEE118": 1.0}
RESULT_FILES = {
    "within": "within_grid.csv",
    "cross": "cross_context.csv",
    "ood": "ood.csv",
}

_ACTIVE_PROCESSES = []
_SIGNAL_RECEIVED = 0


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _job_cost(model, arm, fold=None):
    seconds = MODEL_SECONDS.get(model, 1.0)
    load = 2.0 if arm in ("within", "cross") else 2.0 - OOD_LOAD[fold]
    return seconds * load


def build_jobs(args: argparse.Namespace) -> list[dict[str, object]]:
    """Build and longest-first sort the campaign's independent jobs."""
    jobs = []
    for arm in args.arms:
        if arm not in RESULT_FILES:
            raise ValueError(f"unknown arm: {arm}")
        data_dir = args.data_a if arm == "within" else args.data_b
        for model in args.models:
            seeds = args.seeds_nnconv if model == "nnconv" else args.seeds
            for seed in seeds:
                folds = FOLDS if arm == "ood" else (None,)
                for fold in folds:
                    if arm == "ood":
                        job_id = f"ood_{model}_s{seed}_{fold}"
                    else:
                        job_id = f"{arm}_{model}_s{seed}"
                    out_dir = os.path.join(args.out, job_id)
                    checkpoint_dir = os.path.join(args.ckpt, f"{arm}_{model}")
                    cmd = [
                        sys.executable, "-u", "experiments.py",
                        "--experiment", arm,
                        "--data_dir", data_dir,
                        "--models", model,
                        "--seeds", str(seed),
                        "--epochs", str(args.epochs),
                        "--arch_config", "configs/arch_config.json",
                        "--normalize", "pu_zscore",
                        "--skip_mmd",
                    ]
                    if arm == "within":
                        cmd.extend(("--batch_size", "32", "--regime_tag", "A"))
                    elif arm == "cross":
                        cmd.extend(("--batch_size", "32", "--regime_tag", "B"))
                    else:
                        cmd.extend((
                            "--batch_size_ood", "96",
                            "--regime_tag", "B",
                            "--held_out", fold,
                        ))
                    cmd.extend((
                        "--save_models", checkpoint_dir,
                        "--skip_existing",
                        "--out", out_dir,
                    ))
                    jobs.append({
                        "job_id": job_id,
                        "arm": arm,
                        "cmd": cmd,
                        "out_dir": out_dir,
                        "cost": _job_cost(model, arm, fold),
                    })
    jobs.sort(key=lambda job: -job["cost"])
    return jobs


def _marker_path(job, name):
    return os.path.join(job["out_dir"], name)


def _write_json(path, value):
    temporary_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temporary_path, "w") as fh:
            json.dump(value, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.remove(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _append_ledger(path, event, job_id, attempt, pid, exit_code):
    entry = {
        "event": event,
        "job_id": job_id,
        "attempt": attempt,
        "pid": pid,
        "exit_code": exit_code,
        "time": _utc_now(),
    }
    with open(path, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
        fh.flush()


def _handle_signal(signum, _frame):
    global _SIGNAL_RECEIVED
    _SIGNAL_RECEIVED = signum
    for process in tuple(_ACTIVE_PROCESSES):
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)


def _git_head_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _has_expected_outputs(job):
    return all(
        os.path.isfile(os.path.join(job["out_dir"], name))
        for name in ("summary.json", RESULT_FILES[job["arm"]])
    )


def run_pool(
    jobs: list[dict[str, object]],
    pool: int,
    max_retries: int,
    logs_dir: str,
    ledger_path: str,
) -> list[str]:
    """Run jobs with retries; return job IDs that exhausted their attempts."""
    global _SIGNAL_RECEIVED
    if pool < 1:
        raise ValueError("pool must be at least 1")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(ledger_path)), exist_ok=True)
    pending = [job for job in jobs if not os.path.isfile(_marker_path(job, "DONE.json"))]
    attempts = {}
    job_started = {}
    failed_jobs = []
    active = {}
    _SIGNAL_RECEIVED = 0
    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    for signum in previous_handlers:
        signal.signal(signum, _handle_signal)

    try:
        while (pending and not _SIGNAL_RECEIVED) or active:
            while pending and not _SIGNAL_RECEIVED and len(active) < pool:
                job = pending.pop(0)
                job_id = job["job_id"]
                if os.path.isfile(_marker_path(job, "DONE.json")):
                    continue
                os.makedirs(job["out_dir"], exist_ok=True)
                failed_marker = _marker_path(job, "FAILED.json")
                if os.path.exists(failed_marker):
                    os.remove(failed_marker)
                attempt = attempts.get(job_id, 0) + 1
                attempts[job_id] = attempt
                job_started.setdefault(job_id, time.monotonic())
                log_path = os.path.join(logs_dir, f"{job_id}.log")
                log_file = open(log_path, "a", buffering=1)
                command = job["cmd"]
                log_file.write(
                    f"# attempt {attempt} | {_utc_now()} | "
                    f"{shlex.join(command)}\n"
                )
                environment = os.environ.copy()
                environment["OMP_NUM_THREADS"] = "1"
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=REPO_ROOT,
                        env=environment,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                except OSError as exc:
                    log_file.write(f"launcher error: {exc}\n")
                    log_file.close()
                    _append_ledger(
                        ledger_path, "fail", job_id, attempt, None, None
                    )
                    if attempt <= max_retries:
                        pending.append(job)
                    else:
                        failed_jobs.append(job_id)
                        _write_json(failed_marker, {
                            "job_id": job_id,
                            "attempts": attempt,
                            "exit_code": None,
                            "reason": str(exc),
                            "finished_utc": _utc_now(),
                        })
                    continue
                _ACTIVE_PROCESSES.append(process)
                active[job_id] = {
                    "job": job,
                    "attempt": attempt,
                    "process": process,
                    "log_file": log_file,
                    "started": time.monotonic(),
                }
                _append_ledger(
                    ledger_path, "start", job_id, attempt, process.pid, None
                )

            finished = []
            for job_id, state in active.items():
                process = state["process"]
                exit_code = process.poll()
                if exit_code is None:
                    continue
                finished.append(job_id)
                state["log_file"].close()
                _ACTIVE_PROCESSES.remove(process)
                if _SIGNAL_RECEIVED:
                    continue

                job = state["job"]
                attempt = state["attempt"]
                elapsed = time.monotonic() - job_started[job_id]
                outputs_exist = _has_expected_outputs(job)
                if exit_code == 0 and outputs_exist:
                    _write_json(_marker_path(job, "DONE.json"), {
                        "job_id": job_id,
                        "cmd": job["cmd"],
                        "attempts": attempt,
                        "wall_seconds": elapsed,
                        "finished_utc": _utc_now(),
                        "git_head_sha": _git_head_sha(),
                    })
                    if os.path.exists(_marker_path(job, "FAILED.json")):
                        os.remove(_marker_path(job, "FAILED.json"))
                    _append_ledger(
                        ledger_path, "finish", job_id, attempt,
                        process.pid, exit_code,
                    )
                else:
                    reason = (
                        "missing summary.json or arm result CSV"
                        if exit_code == 0 else "non-zero exit"
                    )
                    active[job_id]["failure_reason"] = reason
                    _append_ledger(
                        ledger_path, "fail", job_id, attempt,
                        process.pid, exit_code,
                    )
                    if attempt <= max_retries:
                        pending.append(job)
                    else:
                        failed_jobs.append(job_id)
                        _write_json(_marker_path(job, "FAILED.json"), {
                            "job_id": job_id,
                            "attempts": attempt,
                            "exit_code": exit_code,
                            "reason": reason,
                            "finished_utc": _utc_now(),
                        })
            for job_id in finished:
                active.pop(job_id)
            if active:
                time.sleep(0.1)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        _ACTIVE_PROCESSES.clear()

    if _SIGNAL_RECEIVED:
        raise SystemExit(128 + _SIGNAL_RECEIVED)
    if failed_jobs:
        print("Exhausted jobs:")
        for job_id in failed_jobs:
            print(f"  {job_id}")
    return failed_jobs


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_a", required=True)
    parser.add_argument("--data_b", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--arms", nargs="+", default=["within", "cross", "ood"])
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--seeds_nnconv", nargs="+", type=int,
                        default=list(SEEDS_NNCONV))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--pool", type=int, default=7)
    parser.add_argument("--max_retries", type=int, default=2)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry_run", action="store_true")
    mode.add_argument("--status", action="store_true")
    return parser


def _print_status(jobs):
    for arm in ("within", "cross", "ood"):
        arm_jobs = [job for job in jobs if job["arm"] == arm]
        done = sum(
            os.path.isfile(_marker_path(job, "DONE.json")) for job in arm_jobs
        )
        failed = sum(
            not os.path.isfile(_marker_path(job, "DONE.json"))
            and os.path.isfile(_marker_path(job, "FAILED.json"))
            for job in arm_jobs
        )
        pending = len(arm_jobs) - done - failed
        print(f"{arm}: done={done} failed={failed} pending={pending}")


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        jobs = build_jobs(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.status:
        _print_status(jobs)
        return 0
    if args.dry_run:
        for job in jobs:
            print(
                f"{job['job_id']}\tarm={job['arm']}\tcost={job['cost']:.2f}\t"
                f"{shlex.join(job['cmd'])}"
            )
        print(f"jobs={len(jobs)}")
        return 0
    failed = run_pool(
        jobs,
        args.pool,
        args.max_retries,
        args.logs,
        os.path.join(args.out, "campaign_ledger.jsonl"),
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
