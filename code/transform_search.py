#!/usr/bin/env python3
"""Exhaustive 'transfer' search: build length-N sequences from known good sequences of
length N-2..N+2 by inserting/deleting 1-2 symbols at ANY positions (and cyclic rotations),
evaluate all variants exactly (batched FFT autocorrelation), report best per N.

usage: tools/transform_search.py NMIN NMAX [--extra results/*.tsv] [--top 20]
Writes seeds_extra/n<N>.txt (top variants) for later MTS refinement.
"""
import argparse, glob, itertools, os
import numpy as np


def load(fn, rec):
    for line in open(fn):
        if line.startswith('#') or not line.strip():
            continue
        f = line.split('#')[0].split()
        bits = f[-1]
        if set(bits) <= {'0', '1'} and len(bits) > 3:
            rec.setdefault(len(bits), set()).add(bits)


def energies(X):
    """X: (B,N) array of +-1 -> exact energies (int64) via FFT autocorrelation."""
    B, N = X.shape
    L = 1 << int(np.ceil(np.log2(2 * N)))
    F = np.fft.rfft(X, L, axis=1)
    ac = np.fft.irfft(F * np.conj(F), L, axis=1)[:, 1:N]
    C = np.rint(ac).astype(np.int64)
    return (C * C).sum(axis=1)


def variants(src, N):
    L = len(src)
    s = np.array([1 if c == '1' else -1 for c in src], dtype=np.int8)
    out = []
    if L == N:  # rotations
        for r in range(1, N):
            out.append(np.roll(s, r))
    elif L == N + 1:
        for i in range(L):
            out.append(np.delete(s, i))
    elif L == N + 2:
        for i in range(L):
            for j in range(i + 1, L):
                out.append(np.delete(s, [i, j]))
    elif L == N - 1:
        for i in range(L + 1):
            for v in (1, -1):
                out.append(np.insert(s, i, v))
    elif L == N - 2:
        for i in range(L + 1):
            for j in range(i, L + 1):
                for a, b in itertools.product((1, -1), repeat=2):
                    out.append(np.insert(s, [i, j], [a, b]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('nmin', type=int)
    ap.add_argument('nmax', type=int)
    ap.add_argument('--records', default='data/records_67_140.tsv')
    ap.add_argument('--extra', nargs='*', default=[])
    ap.add_argument('--top', type=int, default=20)
    a = ap.parse_args()
    rec = {}
    # record table: sequence is column 5
    for line in open(a.records):
        if line.startswith('#'):
            continue
        p = line.rstrip('\n').split('\t')
        if p[4] != '-':
            rec.setdefault(int(p[0]), set()).add(p[4])
    best_known = {}
    for line in open(a.records):
        if not line.startswith('#'):
            p = line.split('\t'); best_known[int(p[0])] = int(p[1])
    for fn in a.extra:
        load(fn, rec)
    os.makedirs('seeds_extra', exist_ok=True)
    for N in range(a.nmin, a.nmax + 1):
        cands = []
        for d in (-2, -1, 0, 1, 2):
            for src in rec.get(N + d, ()):
                cands.extend(variants(src, N))
        if not cands:
            continue
        X = np.array(cands, dtype=np.float64)
        E = energies(X)
        order = np.argsort(E, kind='stable')
        seen, top = set(), []
        for idx in order:
            bits = ''.join('1' if v > 0 else '0' for v in X[idx])
            if bits in seen:
                continue
            seen.add(bits)
            top.append((int(E[idx]), bits))
            if len(top) >= a.top:
                break
        with open(f'seeds_extra/n{N}.txt', 'w') as f:
            for e, bits in top:
                f.write(f'{bits}  # E={e} transfer\n')
        rk = best_known.get(N)
        flag = '  <-- BETTER THAN RECORD' if rk and top[0][0] < rk else ''
        print(f'N={N} variants={len(cands)} best={top[0][0]} record={rk}{flag}', flush=True)


if __name__ == '__main__':
    main()
