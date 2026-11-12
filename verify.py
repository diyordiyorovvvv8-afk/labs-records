#!/usr/bin/env python3
"""Stand-alone LABS verifier (no dependencies).
usage: python3 verify.py [file ...]   (default: N114_E761.txt)
Each non-comment line: a sequence over {0,1} or {-,+} (1/+ = +1, 0/- = -1).
Prints N, energy E = sum_k C_k^2, merit factor F = N^2/(2E), and peak sidelobe level."""
import sys
for fn in sys.argv[1:] or ["N114_E761.txt"]:
    for line in open(fn):
        s = line.split("#")[0].strip()
        if not s:
            continue
        x = [1 if c in "1+" else -1 for c in s if c in "01+-"]
        N = len(x)
        C = [sum(x[i] * x[i + k] for i in range(N - k)) for k in range(1, N)]
        E = sum(c * c for c in C)
        print(f"N={N}  E={E}  F={N * N / (2 * E):.4f}  PSL={max(abs(c) for c in C)}")
