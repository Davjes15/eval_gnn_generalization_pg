#!/usr/bin/env python3
"""Compare the working tree with what a branch on GitHub actually contains.

`.push_branch.py` commits an explicit list of paths, so a file edited locally but
left off the argument list silently never reaches the branch -- which is how the
third audit found documents citing uncommitted artifacts. This lists, for a
branch: remote files that differ from local, remote files missing locally, and
local files absent from the branch.

Usage:
    python3 .diff_branch.py <branch>
"""
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

REPO = "Davjes15/eval_gnn_generalization_pg"
API = f"https://api.github.com/repos/{REPO}"
SKIP_DIRS = {"__pycache__", ".git", "data_a", "data_full", "data_full_v2",
             "ckpt_a", "ckpt_b", "ckpt_norm", "full_run", "transmission",
             "logs", "results_norm_pilot"}
SKIP_EXT = (".pt", ".png", ".mat", ".pyc", ".ipynb_checkpoints")


def token():
    for key in ("GITHUB_PAT", "GITHUB_PAT_TEMP"):
        value = os.environ.get(key)
        if not value:
            continue
        r = urllib.request.Request(API)
        r.add_header("Authorization", f"Bearer {value}")
        try:
            urllib.request.urlopen(r)
            return value
        except urllib.error.HTTPError:
            continue
    raise SystemExit("no usable GitHub token")


def get(url, tok):
    r = urllib.request.Request(url)
    r.add_header("Authorization", f"Bearer {tok}")
    return json.load(urllib.request.urlopen(r))


def blob_sha(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main():
    branch = sys.argv[1]
    tok = token()
    head = get(f"{API}/git/ref/heads/{branch}", tok)["object"]["sha"]
    tree = get(f"{API}/git/trees/{head}?recursive=1", tok)
    if tree.get("truncated"):
        raise SystemExit("tree truncated; refine the comparison")
    remote = {t["path"]: t["sha"] for t in tree["tree"] if t["type"] == "blob"}

    differ, absent = [], []
    for path, sha in sorted(remote.items()):
        if not os.path.exists(path):
            absent.append(path)
        elif blob_sha(path) != sha:
            differ.append(path)

    extra = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            path = os.path.relpath(os.path.join(root, name), ".")
            if path.startswith(".") or path.endswith(SKIP_EXT):
                continue
            if path not in remote:
                extra.append(path)

    print(f"{branch} @ {head[:8]}: {len(remote)} blobs")
    for title, paths in (("LOCAL DIFFERS FROM BRANCH", differ),
                         ("ON BRANCH, MISSING LOCALLY", absent),
                         ("LOCAL ONLY, NEVER PUSHED", sorted(extra))):
        print(f"\n{title} ({len(paths)})")
        for path in paths:
            print("  ", path)


if __name__ == "__main__":
    main()
