# A new best-known low-autocorrelation binary sequence for N = 114

**Date:** 12 November 2026  
**Author:** Diyorbek Madaminov. Search and verification code prepared with the assistance of Claude (Anthropic).

## Result

| | N | Energy E | Merit factor F = N²/(2E) | PSL |
|---|---|---|---|---|
| Previous best-known (Zhang, Shen, Kumar & Pistoia, arXiv:2504.00987v2, 2025) | 114 | 777 | 8.3629 | |
| **This sequence** | **114** | **761** | **6498/761 ≈ 8.5388** | **7** |

Sequence (`1` = +1, `0` = −1), also in `N114_E761.txt`:

```
011010000001001001011010100100100100111111101011101100110011001000101011111100001111010101000011100111001110001111
```

Same sequence as `+`/`−`:

```
-++-+------+--+--+-++-+-+--+--+--+--+++++++-+-+++-++--++--++--+---+-+-++++++----++++-+-+-+----+++--+++--+++---++++
```

Hexadecimal (bit string left-padded with zeros to a multiple of 4 bits): `1a0496a493faeccc8afc3d50e738f`

Aperiodic autocorrelations C_1 … C_113:

```
-1 -2 1 -2 -1 -2 1 -2 -5 0 3 0 1 -2 -3 4 -1 -2 -1 -6 -1 -2 -1 2 1 4 -7 2 3 -2 5 2 -1 -2 5 -4 -3 -2 -3 0 -1 -2 -3 2 3 0 -3 0 -3 -4 1 -4 3 2 1 0 3 6 -3 0 -1 -2 5 2 -3 -2 -3 -2 3 0 1 4 -1 2 -3 0 1 -2 1 0 3 0 1 0 3 0 -3 -2 -7 0 -1 0 3 0 -1 -4 -1 -2 1 0 -3 2 -1 -2 -5 -4 -3 0 3 0 1 0 -1
```

## How it was obtained

The sequence is the N = 112 sequence of Zhang et al. (2025, E = 728) with the two symbols (−1, +1) **prepended**.
It was found by an exhaustive search over all insertions and deletions of 1–3 symbols at the ends of published
best-known sequences of lengths N−3 … N+3 (`code/seeds.py`). A wider search, inserting or deleting 1–2 symbols at **any**
position plus all cyclic rotations, covered about 1.6 million variants for N = 92 … 140 (`code/transform_search.py`). It found no
further improvement. A short memetic tabu search refinement (`code/labs_mts.cpp`, 8 minutes on 4 CPU cores) did not
improve it either.

## Verification

```
python3 verify.py N114_E761.txt
# N=114  E=761  F=8.5388  PSL=7
```

## Literature checked (as of 12 November 2026)

| Source | Best value for N = 114 |
|---|---|
| Knauer table (2004), via the Bošković–Brglez–Brest `git_labs` archive | E = 817 |
| Zhang, Shen, Kumar & Pistoia, arXiv:2504.00987v2 (2025) | E = 777 |
| Discovery@home LABS frontier and candidates (October 2026) | no improvement (best single-bit transfer: E = 785) |
| Dimitrov, arXiv:2206.12070 (2022) | tables start at N = 172 |
| Bošković, Herzog & Brest, arXiv:2210.15962 / J. Comput. Sci. (2024) | odd lengths 171–247 only |
| Pšeničnik et al., arXiv:2409.07222 (2024), arXiv:2607.09688 (2026) | N ≥ 450 only |

We did not find any published sequence of length 114 with E < 777. Corrections are welcome.

## Files

- `N114_E761.txt`: the sequence
- `verify.py`: stand-alone verifier (pure Python)
- `code/`: search code (`seeds.py`, `transform_search.py`, `labs_mts.cpp`: multithreaded memetic tabu search)
