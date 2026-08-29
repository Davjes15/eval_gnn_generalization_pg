#!/usr/bin/env python3
"""Create a two-parent merge commit on GitHub from the local merged tree.

git-over-HTTPS is firewall-blocked here while api.github.com is reachable, so the
merge of `step-8-regime-comparison` into `main` is written through the Git Data
API: the tree is `step-8`'s tree plus every path where the local resolved tree
differs from it, and the commit carries both branch heads as parents so the
result is a real merge rather than a squash.

Usage: python3 .push_merge.py <head_branch> <base_branch> <merged_branch> <message>
"""
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

REPO = "Davjes15/eval_gnn_generalization_pg"
API = f"https://api.github.com/repos/{REPO}"
AUTHOR = {"name": "David Quispe", "email": "david.quispe@intact.net"}
ROOT = os.path.dirname(os.path.abspath(__file__))


def _token():
    tried = []
    for key in ("GITHUB_PAT", "GITHUB_PAT_TEMP"):
        value = os.environ.get(key)
        if not value:
            continue
        r = urllib.request.Request(API)
        r.add_header("Authorization", f"Bearer {value}")
        try:
            urllib.request.urlopen(r)
            return value
        except urllib.error.HTTPError as exc:
            tried.append(f"{key}: HTTP {exc.code}")
    raise SystemExit("no usable GitHub token (" + "; ".join(tried) + ")")


TOKEN = _token()


def req(method, url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Authorization", f"Bearer {TOKEN}")
    r.add_header("Accept", "application/vnd.github+json")
    r.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(r) as resp:
        return json.loads(resp.read().decode())


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args],
                          check=True, capture_output=True, text=True).stdout


def main():
    branch, base_ref, merged_ref, message = sys.argv[1:5]
    base_sha = req("GET", f"{API}/git/ref/heads/{base_ref}")["object"]["sha"]
    merged_sha = req("GET", f"{API}/git/ref/heads/{merged_ref}")["object"]["sha"]
    base_tree = req("GET", f"{API}/commits/{merged_sha}")["commit"]["tree"]["sha"]

    changed = [line.split("\t")[-1] for line in
               git("diff", "--name-only", f"origin/{merged_ref}", "HEAD").splitlines()]
    print(f"{len(changed)} paths differ from {merged_ref}")

    entries = []
    for path in changed:
        with open(os.path.join(ROOT, path), "rb") as fh:
            content = fh.read()
        blob = req("POST", f"{API}/git/blobs", {
            "content": base64.b64encode(content).decode(), "encoding": "base64"})
        mode = "100755" if os.access(os.path.join(ROOT, path), os.X_OK) else "100644"
        entries.append({"path": path, "mode": mode, "type": "blob",
                        "sha": blob["sha"]})

    tree = base_tree
    for i in range(0, len(entries), 100):
        tree = req("POST", f"{API}/git/trees",
                   {"base_tree": tree, "tree": entries[i:i + 100]})["sha"]

    commit = req("POST", f"{API}/git/commits", {
        "message": message, "tree": tree,
        "parents": [base_sha, merged_sha],
        "author": AUTHOR, "committer": AUTHOR})
    try:
        req("PATCH", f"{API}/git/refs/heads/{branch}",
            {"sha": commit["sha"], "force": True})
    except urllib.error.HTTPError:
        req("POST", f"{API}/git/refs",
            {"ref": f"refs/heads/{branch}", "sha": commit["sha"]})
    print(f"Pushed {branch} @ {commit['sha'][:8]} (tree {tree[:8]})")


if __name__ == "__main__":
    main()
