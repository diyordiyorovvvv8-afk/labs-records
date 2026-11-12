#!/usr/bin/env python3
"""Generate seed sequences for a target length N from best-known records at nearby lengths.

Operations (as used by the 2026 'endpoint edit' candidates): take the record at length N+d
and delete |d| bits from the ends (d>0), or append |d| bits at the ends (d<0, all sign
patterns).  Also include the record at N itself.  Output: one 0/1 string per line,
sorted by energy, with a comment.

usage: tools/seeds.py N [--records data/records_67_140.tsv] [--span 3] [--top 60]
"""
import argparse, itertools, sys


def energy(bits):
    s = [1 if b == '1' else -1 for b in bits]
    n = len(s)
    return sum(sum(s[i] * s[i + k] for i in range(n - k)) ** 2 for k in range(1, n))


def load_records(fn):
    rec = {}
    for line in open(fn):
        if line.startswith('#') or not line.strip():
            continue
        p = line.rstrip('\n').split('\t')
        n, e, seq = int(p[0]), int(p[1]), p[4]
        if seq != '-':
            rec.setdefault(n, []).append((e, seq, p[3]))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('N', type=int)
    ap.add_argument('--records', default='data/records_67_140.tsv')
    ap.add_argument('--extra', action='append', default=[],
                    help='extra file(s) with 0/1 sequences of any length (e.g. our own results)')
    ap.add_argument('--span', type=int, default=3)
    ap.add_argument('--top', type=int, default=60)
    a = ap.parse_args()
    N = a.N
    rec = load_records(a.records)
    for fn in a.extra:  # lines whose last whitespace-separated field is a 0/1 string
        for line in open(fn):
            f = line.split('#')[0].split()
            if f and set(f[-1]) <= {'0', '1'}:
                rec.setdefault(len(f[-1]), []).append((energy(f[-1]), f[-1], fn))
    cands = {}

    def add(bits, how):
        if len(bits) != N:
            return
        if bits not in cands:
            cands[bits] = (energy(bits), how)

    for d in range(-a.span, a.span + 1):
        for e0, seq, src in rec.get(N + d, []):
            tag = f'N={N + d} E={e0} ({src})'
            if d == 0:
                add(seq, 'record ' + tag)
            elif d > 0:  # delete d bits: a from the left, d-a from the right
                for left in range(d + 1):
                    add(seq[left:len(seq) - (d - left)], f'del L{left} R{d - left} from {tag}')
            else:  # append -d bits: a on the left, rest on the right, all sign patterns
                m = -d
                for left in range(m + 1):
                    for pat in itertools.product('01', repeat=m):
                        pat = ''.join(pat)
                        add(pat[:left] + seq + pat[left:], f'add L{pat[:left] or "-"} R{pat[left:] or "-"} to {tag}')
    best = sorted(cands.items(), key=lambda kv: kv[1][0])[:a.top]
    for bits, (e, how) in best:
        print(f'{bits}  # E={e} F={N * N / (2 * e):.4f} {how}')
    if best:
        print(f'# N={N}: {len(cands)} distinct candidates, best E={best[0][1][0]}', file=sys.stderr)


if __name__ == '__main__':
    main()
