"""Folder-name normalization for feasibility reference rows."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from summarize_feasibility import regime_label


def main():
    cases = (
        ("data_regime_a_fixed_topology", "regime_a"),
        ("data_regime_a_fixed_topology_v3", "regime_a"),
        ("data_regime_b_contingencies_v3", "regime_b"),
        ("path/to/data_regime_b_contingencies_v3/", "regime_b"),
        ("custom_dataset", "custom_dataset"),
    )
    failures = []
    for data_dir, expected in cases:
        actual = regime_label(data_dir)
        result = "PASS" if actual == expected else "FAIL"
        print(f"  [{result}] {data_dir!r} -> {actual!r}; expected {expected!r}")
        if actual != expected:
            failures.append(data_dir)

    print("\nALL CHECKS PASSED" if not failures
          else f"\nFAILURES: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
