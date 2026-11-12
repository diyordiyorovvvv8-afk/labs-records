// labs_mts.cpp -- Memetic Tabu Search (MTS) for the LABS problem.
//
//   Energy  E(s) = sum_{k=1}^{N-1} C_k^2,   C_k = sum_{i=0}^{N-1-k} s_i s_{i+k},   s_i in {+1,-1}
//   Merit factor F = N^2 / (2E)
//
// Algorithm: Gallardo, Cotta & Fernandez (2009) memetic algorithm + tabu search,
// as also used (on GPU) by Zhang et al. 2025 (arXiv:2504.00987).  Each thread runs an
// independent memetic "island".  Optional skew-symmetric mode for odd N.
//
// Neighbourhood evaluation (all N single flips) in O(N^2) per step with SIMD-friendly
// int16 dot products:
//     dE_i = -4 s_i G_i + 4(N-1) + 8 P_i
//     G_i  = sum_{m != i} C_{|m-i|} s_m          (Toeplitz mat-vec, contiguous dot product)
//     P_i  = sum_{k=1}^{min(i,N-1-i)} s_{i+k} s_{i-k}   (maintained incrementally, O(N) per flip)
//
// Build:  g++ -O3 -march=native -std=c++17 -pthread labs_mts.cpp -o labs_mts
// Usage:  ./labs_mts -n 114 -t 4 --time 3600 --out results/n114.tsv [--init seeds.txt] [--skew]
//         ./labs_mts --selftest
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using u64 = uint64_t;

// ---------------------------------------------------------------- RNG (xoshiro256**)
struct RNG {
    u64 s[4];
    static u64 splitmix(u64 &x) {
        u64 z = (x += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    explicit RNG(u64 seed = 1) { for (int i = 0; i < 4; i++) s[i] = splitmix(seed); }
    static inline u64 rotl(u64 x, int k) { return (x << k) | (x >> (64 - k)); }
    inline u64 next() {
        u64 r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
        return r;
    }
    inline u64 below(u64 n) { return (u64)(((__uint128_t)next() * n) >> 64); }  // [0,n)
    inline int range(int a, int b) { return a + (int)below((u64)(b - a + 1)); }  // [a,b]
    inline double uni() { return (double)(next() >> 11) * 0x1.0p-53; }
};

// ---------------------------------------------------------------- helpers
static i64 energy_of(const i16 *s, int N) {
    i64 E = 0;
    for (int k = 1; k < N; k++) {
        i64 c = 0;
        for (int i = 0; i + k < N; i++) c += s[i] * s[i + k];
        E += c * c;
    }
    return E;
}
static std::string to_bits(const i16 *s, int N) {
    std::string r(N, '0');
    for (int i = 0; i < N; i++) r[i] = s[i] > 0 ? '1' : '0';
    return r;
}
static double now_sec() {
    using namespace std::chrono;
    static const auto t0 = steady_clock::now();
    return duration<double>(steady_clock::now() - t0).count();
}
// skew-symmetry (odd N = 2c+1, 0-indexed centre c):  s[c+i] = (-1)^i s[c-i]
static void make_skew(i16 *s, int N) {
    int c = (N - 1) / 2;
    for (int i = 1; i <= c; i++) s[c + i] = (i & 1) ? -s[c - i] : s[c - i];
}
static bool is_skew(const i16 *s, int N) {
    if (!(N & 1)) return false;
    int c = (N - 1) / 2;
    for (int i = 1; i <= c; i++)
        if (s[c + i] != ((i & 1) ? -s[c - i] : s[c - i])) return false;
    return true;
}

// ---------------------------------------------------------------- incremental state
struct State {
    int N = 0;
    std::vector<i16> s;   // sequence (+1/-1)
    std::vector<i32> C;   // C[k], k=1..N-1
    std::vector<i16> CC;  // 2N-1 entries: CC[N-1+d] = C[|d|], CC[N-1] = 0
    std::vector<i32> P;   // P[i]
    std::vector<i32> dE;  // neighbour deltas
    std::vector<i16> pad; // 3N, pad[N+x] = s[x], zeros outside (skew evaluation)
    i64 E = 0;

    void init(int n) {
        N = n;
        s.assign(N, 1); C.assign(N, 0); CC.assign(2 * N - 1, 0); P.assign(N, 0);
        dE.assign(N, 0); pad.assign(3 * N, 0);
    }
    void load(const i16 *seq) {
        for (int i = 0; i < N; i++) s[i] = seq[i];
        E = 0;
        for (int k = 1; k < N; k++) {
            int c = 0;
            for (int i = 0; i + k < N; i++) c += s[i] * s[i + k];
            C[k] = c;
            E += (i64)c * c;
        }
        refresh_CC();
        for (int i = 0; i < N; i++) {
            int p = 0, L = std::min(i, N - 1 - i);
            for (int k = 1; k <= L; k++) p += s[i + k] * s[i - k];
            P[i] = p;
        }
        for (int i = 0; i < N; i++) pad[N + i] = s[i];
    }
    inline void refresh_CC() {
        CC[N - 1] = 0;
        for (int k = 1; k < N; k++) CC[N - 1 + k] = CC[N - 1 - k] = (i16)C[k];
    }
    // all single-flip deltas -> dE[0..N-1]
    void eval_all() {
        const i16 *__restrict cc = CC.data();
        const i16 *__restrict sv = s.data();
        const int base = 4 * (N - 1);
        for (int i = 0; i < N; i++) {
            const i16 *__restrict w = cc + (N - 1 - i);
            int g = 0;
            for (int m = 0; m < N; m++) g += (int)w[m] * (int)sv[m];
            dE[i] = -4 * sv[i] * g + base + 8 * P[i];
        }
    }
    // flip one bit, O(N)
    void flip(int j) {
        const int sj = s[j];
        const int m1 = N - 1 - j;
        for (int k = 1; k <= m1; k++) C[k] -= 2 * sj * s[j + k];
        for (int k = 1; k <= j; k++) C[k] -= 2 * sj * s[j - k];
        for (int d = 1; j + 2 * d < N; d++) P[j + d] -= 2 * sj * s[j + 2 * d];
        for (int d = 1; j - 2 * d >= 0; d++) P[j - d] -= 2 * sj * s[j - 2 * d];
        s[j] = (i16)-sj;
        pad[N + j] = (i16)-sj;
        i64 e = 0;
        for (int k = 1; k < N; k++) e += (i64)C[k] * C[k];
        E = e;
        refresh_CC();
    }
    // ---- skew-symmetric moves: m=0 flips centre c, m>=1 flips pair (c-m, c+m)
    i64 skew_delta(int m) const {
        const int c = (N - 1) / 2;
        const i16 *__restrict pd = pad.data() + N;
        i64 de = 0;
        if (m == 0) {
            const int sp = s[c];
            for (int k = 2; k < N; k += 2) {
                int d = -2 * sp * (pd[c + k] + pd[c - k]);
                de += (i64)(2 * C[k] + d) * d;
            }
        } else {
            const int p = c - m, q = c + m, sp = s[p], sq = s[q];
            for (int k = 2; k < N; k += 2) {
                int d = -2 * sp * (pd[p + k] + pd[p - k]) - 2 * sq * (pd[q + k] + pd[q - k]);
                if (k == 2 * m) d += 4 * sp * sq;
                de += (i64)(2 * C[k] + d) * d;
            }
        }
        return de;
    }
    void skew_apply(int m) {
        const int c = (N - 1) / 2;
        if (m == 0) flip(c);
        else { flip(c - m); flip(c + m); }
    }
};

// ---------------------------------------------------------------- parameters
struct Params {
    int N = 0;
    int threads = 1;
    double time_limit = 60;
    i64 target = -1;
    u64 seed = 0;
    int pop = 100;
    double pcomb = 0.9;
    double pmut = -1;  // default 1/N (or 1/nfree in skew mode)
    double iter_lo = 0.5, iter_hi = 1.5;  // tabu iterations ~ U[lo*n, hi*n]
    double tabu_lo = 0.1, tabu_hi = 0.12; // tenure ~ U[lo*maxIter, hi*maxIter]
    bool skew = false;
    bool replace_worst = true;
    bool aspiration = true;
    std::string out, init;
    int init_threads = -1;  // how many threads seed populations from --init (default all)
    int init_max = 50;      // max seeds per population
    double report = 30;
    bool quiet = false;
};

// ---------------------------------------------------------------- shared best
struct Shared {
    std::mutex mu;
    std::atomic<i64> bestE{INT64_MAX};
    std::atomic<bool> stop{false};
    std::vector<i16> best;
    std::atomic<u64> tabu_runs{0}, flips{0};
    FILE *out = nullptr;
};

static void report_improvement(Shared &sh, const Params &P, const i16 *s, i64 E, int tid) {
    if (E >= sh.bestE.load(std::memory_order_relaxed)) return;
    std::lock_guard<std::mutex> lk(sh.mu);
    if (E >= sh.bestE.load()) return;
    i64 chk = energy_of(s, P.N);  // independent recomputation
    if (chk != E) {
        fprintf(stderr, "INTERNAL ERROR: incremental E=%" PRId64 " but recomputed E=%" PRId64 "\n", E, chk);
        std::abort();
    }
    sh.bestE.store(E);
    sh.best.assign(s, s + P.N);
    double F = (double)P.N * P.N / (2.0 * E);
    std::string bits = to_bits(s, P.N);
    double t = now_sec();
    if (!P.quiet) {
        printf("[%9.2fs] thr %2d  N=%d  E=%" PRId64 "  F=%.4f  skew=%d  %s\n", t, tid, P.N, E, F,
               (int)is_skew(s, P.N), bits.c_str());
        fflush(stdout);
    }
    if (sh.out) {
        fprintf(sh.out, "%d\t%" PRId64 "\t%.6f\t%.2f\t%d\t%s\n", P.N, E, F, t, tid, bits.c_str());
        fflush(sh.out);
    }
    if (P.target >= 0 && E <= P.target) sh.stop.store(true);
}

// ---------------------------------------------------------------- one memetic island
struct Island {
    const Params &P;
    Shared &sh;
    int tid;
    RNG rng;
    int N, nfree;
    State st;
    std::vector<std::vector<i16>> pop;
    std::vector<i64> popE;
    std::vector<i16> child, cur_best;
    std::vector<int> tabu;
    std::vector<i64> mdelta;  // skew move deltas

    Island(const Params &p, Shared &s, int t, u64 seed)
        : P(p), sh(s), tid(t), rng(seed), N(p.N) {
        nfree = P.skew ? (N + 1) / 2 : N;
        st.init(N);
        child.assign(N, 1); cur_best.assign(N, 1);
        tabu.assign(N, 0);
        mdelta.assign(nfree, 0);
    }

    void random_seq(i16 *s) {
        for (int i = 0; i < N; i++) s[i] = (rng.next() >> 63) ? 1 : -1;
        if (P.skew) make_skew(s, N);
    }

    // tabu search from `seq`; result written back into seq, returns its energy
    i64 tabu_search(i16 *seq) {
        st.load(seq);
        i64 bestE = st.E;
        std::copy(st.s.begin(), st.s.end(), cur_best.begin());
        int maxIter = rng.range(std::max(1, (int)(P.iter_lo * nfree)), std::max(1, (int)(P.iter_hi * nfree)));
        int tmin = std::max(1, (int)(P.tabu_lo * maxIter));
        int tmax = std::max(tmin, (int)(P.tabu_hi * maxIter));
        std::fill(tabu.begin(), tabu.begin() + nfree, 0);
        for (int t = 1; t <= maxIter; t++) {
            int jbest = -1, cnt = 0;
            i64 vbest = INT64_MAX;
            if (!P.skew) {
                st.eval_all();
                for (int i = 0; i < N; i++) {
                    i64 v = st.dE[i];
                    bool ok = tabu[i] < t || (P.aspiration && st.E + v < bestE);
                    if (!ok) continue;
                    if (v < vbest) { vbest = v; jbest = i; cnt = 1; }
                    else if (v == vbest && rng.below(++cnt) == 0) jbest = i;
                }
                if (jbest < 0) continue;
                st.flip(jbest);
            } else {
                for (int m = 0; m < nfree; m++) {
                    i64 v = st.skew_delta(m);
                    bool ok = tabu[m] < t || (P.aspiration && st.E + v < bestE);
                    if (!ok) continue;
                    if (v < vbest) { vbest = v; jbest = m; cnt = 1; }
                    else if (v == vbest && rng.below(++cnt) == 0) jbest = m;
                }
                if (jbest < 0) continue;
                st.skew_apply(jbest);
            }
            tabu[jbest] = t + rng.range(tmin, tmax);
            if (st.E < bestE) {
                bestE = st.E;
                std::copy(st.s.begin(), st.s.end(), cur_best.begin());
            }
        }
        sh.flips.fetch_add((u64)maxIter, std::memory_order_relaxed);
        sh.tabu_runs.fetch_add(1, std::memory_order_relaxed);
        std::copy(cur_best.begin(), cur_best.end(), seq);
        return bestE;
    }

    int tournament() {
        int a = (int)rng.below(pop.size()), b = (int)rng.below(pop.size());
        return popE[a] <= popE[b] ? a : b;
    }

    void run(const std::vector<std::vector<i16>> &seeds) {
        const int K = P.pop;
        pop.assign(K, std::vector<i16>(N));
        popE.assign(K, 0);
        int ns = std::min((int)seeds.size(), std::min(K, P.init_max));
        // spread the seed list over islands: rotate start index by thread id
        for (int k = 0; k < K; k++) {
            if (k < ns) pop[k] = seeds[(k + (size_t)tid * ns) % seeds.size()];
            else random_seq(pop[k].data());
            if (P.skew && !is_skew(pop[k].data(), N)) make_skew(pop[k].data(), N);
            popE[k] = tabu_search(pop[k].data());
            report_improvement(sh, P, pop[k].data(), popE[k], tid);
            if (sh.stop.load(std::memory_order_relaxed)) return;
        }
        const double pm = P.pmut > 0 ? P.pmut : 1.0 / nfree;
        const int c = (N - 1) / 2;
        while (!sh.stop.load(std::memory_order_relaxed)) {
            if (rng.uni() < P.pcomb) {
                const i16 *a = pop[tournament()].data();
                const i16 *b = pop[tournament()].data();
                if (!P.skew) {
                    int cut = rng.range(1, N - 1);
                    for (int i = 0; i < N; i++) child[i] = i < cut ? a[i] : b[i];
                } else {  // crossover on the free half 0..c
                    int cut = rng.range(1, c);
                    for (int i = 0; i <= c; i++) child[i] = i < cut ? a[i] : b[i];
                }
            } else {
                const i16 *a = pop[rng.below(K)].data();
                std::copy(a, a + N, child.begin());
            }
            if (!P.skew) {
                for (int i = 0; i < N; i++) if (rng.uni() < pm) child[i] = -child[i];
            } else {
                for (int i = 0; i <= c; i++) if (rng.uni() < pm) child[i] = -child[i];
                make_skew(child.data(), N);
            }
            i64 e = tabu_search(child.data());
            if (e < sh.bestE.load(std::memory_order_relaxed)) report_improvement(sh, P, child.data(), e, tid);
            // replacement
            int w;
            if (P.replace_worst) {
                w = 0;
                for (int k = 1; k < K; k++) if (popE[k] > popE[w]) w = k;
                if (e >= popE[w]) continue;
                bool dup = false;
                for (int k = 0; k < K && !dup; k++)
                    if (popE[k] == e && std::equal(child.begin(), child.end(), pop[k].begin())) dup = true;
                if (dup) continue;
            } else {
                w = (int)rng.below(K);
            }
            pop[w] = child;
            popE[w] = e;
        }
    }
};

// ---------------------------------------------------------------- seeds file
static std::vector<std::vector<i16>> load_seeds(const std::string &fn, int N) {
    std::vector<std::vector<i16>> r;
    std::ifstream f(fn);
    if (!f) { fprintf(stderr, "cannot open %s\n", fn.c_str()); exit(1); }
    std::string line;
    while (std::getline(f, line)) {
        std::string b;
        for (char ch : line) {
            if (ch == '#') break;
            if (ch == '0' || ch == '1') b.push_back(ch);
            else if (ch == '+') b.push_back('1');
            else if (ch == '-') b.push_back('0');
            else if (ch == '\t' || ch == ' ') { if ((int)b.size() == N) break; b.clear(); }  // last field wins
        }
        if ((int)b.size() != N) continue;
        std::vector<i16> s(N);
        for (int i = 0; i < N; i++) s[i] = b[i] == '1' ? 1 : -1;
        r.push_back(s);
    }
    return r;
}

// ---------------------------------------------------------------- self test
static int selftest() {
    RNG rng(12345);
    int bad = 0;
    for (int trial = 0; trial < 200; trial++) {
        int N = 3 + (int)rng.below(120);
        State st; st.init(N);
        std::vector<i16> s(N);
        for (int i = 0; i < N; i++) s[i] = (rng.next() >> 63) ? 1 : -1;
        st.load(s.data());
        for (int step = 0; step < 50; step++) {
            st.eval_all();
            for (int i = 0; i < N; i++) {  // brute-force check of every delta
                std::vector<i16> t(st.s); t[i] = -t[i];
                i64 want = energy_of(t.data(), N) - st.E;
                if (want != st.dE[i]) { if (bad++ < 5) printf("delta mismatch N=%d i=%d %d vs %" PRId64 "\n", N, i, st.dE[i], want); }
            }
            int j = (int)rng.below(N);
            st.flip(j);
            if (energy_of(st.s.data(), N) != st.E) { if (bad++ < 5) printf("flip E mismatch N=%d\n", N); }
            State ref; ref.init(N); ref.load(st.s.data());
            if (ref.P != st.P || ref.C != st.C) { if (bad++ < 5) printf("P/C mismatch N=%d\n", N); }
        }
        if (N & 1) {  // skew moves
            make_skew(s.data(), N);
            st.load(s.data());
            int nf = (N + 1) / 2;
            for (int step = 0; step < 30; step++) {
                for (int m = 0; m < nf; m++) {
                    std::vector<i16> t(st.s);
                    int c = (N - 1) / 2;
                    if (m == 0) t[c] = -t[c]; else { t[c - m] = -t[c - m]; t[c + m] = -t[c + m]; }
                    if (!is_skew(t.data(), N)) { if (bad++ < 5) printf("skew move breaks symmetry\n"); }
                    i64 want = energy_of(t.data(), N) - st.E;
                    i64 got = st.skew_delta(m);
                    if (want != got) { if (bad++ < 5) printf("skew delta mismatch N=%d m=%d %" PRId64 " vs %" PRId64 "\n", N, m, got, want); }
                }
                st.skew_apply((int)rng.below(nf));
                if (!is_skew(st.s.data(), N) || energy_of(st.s.data(), N) != st.E) { if (bad++ < 5) printf("skew apply mismatch\n"); }
            }
        }
    }
    // Barker 13 must have E=6 and be skew-symmetric
    const char *b13 = "1111100110101";
    std::vector<i16> b(13);
    for (int i = 0; i < 13; i++) b[i] = b13[i] == '1' ? 1 : -1;
    if (energy_of(b.data(), 13) != 6 || !is_skew(b.data(), 13)) { bad++; printf("Barker-13 check failed\n"); }
    printf(bad ? "SELFTEST FAILED (%d errors)\n" : "SELFTEST OK\n", bad);
    return bad ? 1 : 0;
}

// ---------------------------------------------------------------- main
static void usage() {
    printf("labs_mts -n N [-t threads] [--time sec] [--target E] [--seed S] [--pop K]\n"
           "         [--skew] [--init seeds.txt] [--init-threads k] [--init-max m] [--out file.tsv]\n"
           "         [--replace worst|random] [--no-aspiration] [--pmut p] [--pcomb p]\n"
           "         [--iter lo hi] [--tenure lo hi] [--report sec] [--quiet]\n"
           "labs_mts --selftest | --bench N\n");
}

int main(int argc, char **argv) {
    Params P;
    int bench = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nxt = [&]() -> const char * { if (i + 1 >= argc) { usage(); exit(1); } return argv[++i]; };
        if (a == "-n") P.N = atoi(nxt());
        else if (a == "-t") P.threads = atoi(nxt());
        else if (a == "--time") P.time_limit = atof(nxt());
        else if (a == "--target") P.target = atoll(nxt());
        else if (a == "--seed") P.seed = strtoull(nxt(), nullptr, 10);
        else if (a == "--pop") P.pop = atoi(nxt());
        else if (a == "--skew") P.skew = true;
        else if (a == "--init") P.init = nxt();
        else if (a == "--init-threads") P.init_threads = atoi(nxt());
        else if (a == "--init-max") P.init_max = atoi(nxt());
        else if (a == "--out") P.out = nxt();
        else if (a == "--replace") P.replace_worst = std::string(nxt()) == "worst";
        else if (a == "--no-aspiration") P.aspiration = false;
        else if (a == "--pmut") P.pmut = atof(nxt());
        else if (a == "--pcomb") P.pcomb = atof(nxt());
        else if (a == "--iter") { P.iter_lo = atof(nxt()); P.iter_hi = atof(nxt()); }
        else if (a == "--tenure") { P.tabu_lo = atof(nxt()); P.tabu_hi = atof(nxt()); }
        else if (a == "--report") P.report = atof(nxt());
        else if (a == "--quiet") P.quiet = true;
        else if (a == "--selftest") return selftest();
        else if (a == "--bench") bench = atoi(nxt());
        else { usage(); return 1; }
    }
    if (bench) {  // raw neighbourhood-evaluation speed
        int N = bench;
        State st; st.init(N);
        RNG rng(1);
        std::vector<i16> s(N);
        for (int i = 0; i < N; i++) s[i] = (rng.next() >> 63) ? 1 : -1;
        st.load(s.data());
        double t0 = now_sec();
        long steps = 0;
        i64 acc = 0;
        while (now_sec() - t0 < 2.0) {
            for (int r = 0; r < 1000; r++) {
                st.eval_all();
                int j = (int)rng.below(N);
                acc += st.dE[j];
                st.flip(j);
            }
            steps += 1000;
        }
        double dt = now_sec() - t0;
        printf("N=%d: %.3f M steps/s (full neighbourhood eval + flip), %.1f ns/step  [%lld]\n", N,
               steps / dt / 1e6, dt / steps * 1e9, (long long)(acc & 1));
        return 0;
    }
    if (P.N < 3) { usage(); return 1; }
    if (P.skew && !(P.N & 1)) { fprintf(stderr, "--skew requires odd N\n"); return 1; }
    if (P.N > 16000) { fprintf(stderr, "N too large\n"); return 1; }
    if (P.seed == 0) P.seed = (u64)std::chrono::high_resolution_clock::now().time_since_epoch().count();

    Shared sh;
    if (!P.out.empty()) {
        sh.out = fopen(P.out.c_str(), "a");
        if (!sh.out) { fprintf(stderr, "cannot open %s\n", P.out.c_str()); return 1; }
    }
    std::vector<std::vector<i16>> seeds;
    if (!P.init.empty()) {
        seeds = load_seeds(P.init, P.N);
        printf("loaded %zu seed sequences of length %d\n", seeds.size(), P.N);
    }
    int seeded_threads = P.init_threads < 0 ? P.threads : P.init_threads;
    printf("LABS MTS: N=%d threads=%d pop=%d skew=%d time=%.0fs target=%" PRId64 " seed=%" PRIu64
           " replace=%s aspiration=%d\n", P.N, P.threads, P.pop, (int)P.skew, P.time_limit, P.target, P.seed,
           P.replace_worst ? "worst" : "random", (int)P.aspiration);
    fflush(stdout);

    std::vector<std::thread> th;
    std::vector<std::vector<i16>> empty;
    for (int t = 0; t < P.threads; t++) {
        th.emplace_back([&, t]() {
            Island isl(P, sh, t, P.seed * 0x9E3779B97F4A7C15ULL + 7919ULL * (t + 1));
            isl.run(t < seeded_threads ? seeds : empty);
        });
    }
    double t0 = now_sec(), last = t0;
    while (!sh.stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        double t = now_sec();
        if (t - t0 >= P.time_limit) { sh.stop.store(true); break; }
        if (P.report > 0 && t - last >= P.report) {
            last = t;
            printf("  .. %.0fs best E=%" PRId64 " F=%.4f | %.0f tabu-runs/s %.2f M flips/s\n", t - t0,
                   sh.bestE.load(), (double)P.N * P.N / (2.0 * sh.bestE.load()),
                   sh.tabu_runs.load() / (t - t0), sh.flips.load() / (t - t0) / 1e6);
            fflush(stdout);
        }
    }
    for (auto &x : th) x.join();
    double dt = now_sec() - t0;
    i64 E = sh.bestE.load();
    printf("DONE N=%d bestE=%" PRId64 " F=%.4f time=%.2fs tabu_runs=%" PRIu64 " flips=%" PRIu64 "\n", P.N, E,
           (double)P.N * P.N / (2.0 * E), dt, sh.tabu_runs.load(), sh.flips.load());
    if (!sh.best.empty()) printf("BEST %s\n", to_bits(sh.best.data(), P.N).c_str());
    if (sh.out) fclose(sh.out);
    return 0;
}
