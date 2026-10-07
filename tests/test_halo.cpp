// ranks: 1 2 4
//
// test_halo -- Spec "Correctness bar": fill phi with a known global function,
// exchange, compare every guard cell against the analytic value on 1/2/4 ranks.
//
// The expected value in a guard cell is the global function evaluated at the
// *clamped* global cell index: unclamped across a rank boundary (the exchange
// must reproduce the serial value exactly), clamped at a physical wall (which
// is exactly what zero-gradient Neumann extrapolation gives).  Corner guard
// cells are clamped in whichever direction touches a wall, which is what the
// x-then-y sweep must produce.

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "grid.h"
#include "params.h"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSentinel = -1.0e30;  // guard cells start poisoned

// A smooth, non-symmetric global function; no symmetry that could hide a
// transposed index or an off-by-one in the guard depth.
double gfun(double x, double y) {
    return std::sin(2.0 * kPi * x) * std::cos(3.0 * kPi * y) + 0.37 * x - 0.21 * y + 1.0;
}

int clampIndex(int i, int n) {
    if (i < 0) return 0;
    if (i > n - 1) return n - 1;
    return i;
}

// Returns the max-norm guard-cell error for one decomposition, or -1 on setup
// failure.
double runCase(int nx, int ny, int px, int py, int rank, std::string& err) {
    Params p;
    p.nx = nx;
    p.ny = ny;
    p.xmin = 0.0;
    p.xmax = 1.0;
    p.ymin = 0.0;
    p.ymax = 1.0;
    p.px = px;
    p.py = py;

    Grid g;
    if (!g.create(p, MPI_COMM_WORLD, err)) return -1.0;

    std::vector<double> f = g.allocate(kSentinel);

    // owned cells only -- everything else must come from the exchange
    for (int j = 0; j < g.ny(); ++j) {
        for (int i = 0; i < g.nx(); ++i) {
            f[g.index(i, j)] = gfun(g.xCenter(i), g.yCenter(j));
        }
    }

    g.exchange(f);

    double maxErr = 0.0;
    int nUnfilled = 0;
    for (int j = -g.ng(); j < g.ny() + g.ng(); ++j) {
        for (int i = -g.ng(); i < g.nx() + g.ng(); ++i) {
            const double got = f[g.index(i, j)];
            if (got == kSentinel) {
                ++nUnfilled;
                continue;
            }
            const int gi = clampIndex(g.i0() + i, g.nxGlobal());
            const int gj = clampIndex(g.j0() + j, g.nyGlobal());
            const double x = p.xmin + (static_cast<double>(gi) + 0.5) * g.dx();
            const double y = p.ymin + (static_cast<double>(gj) + 0.5) * g.dy();
            const double want = gfun(x, y);
            const double e = std::fabs(got - want);
            if (e > maxErr) maxErr = e;
        }
    }

    if (nUnfilled > 0) {
        err = "rank " + std::to_string(rank) + ": " + std::to_string(nUnfilled) +
              " guard cells were never filled";
        g.destroy();
        return -1.0;
    }

    double globalErr = 0.0;
    MPI_Allreduce(&maxErr, &globalErr, 1, MPI_DOUBLE, MPI_MAX, g.comm());
    g.destroy();
    return globalErr;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Exercise the automatic decomposition and both degenerate ones, so the
    // x sweep, the y sweep and their corners are all covered.
    struct Case {
        const char* name;
        int px;
        int py;
    };
    const Case cases[3] = {{"dims_create", 0, 0}, {"px=nprocs", nprocs, 1}, {"py=nprocs", 1, nprocs}};

    const double tol = 1.0e-12;
    int failures = 0;

    for (const Case& c : cases) {
        std::string err;
        const double e = runCase(96, 72, c.px, c.py, rank, err);
        if (e < 0.0) {
            if (rank == 0) std::printf("test_halo: %-12s SETUP FAIL: %s\n", c.name, err.c_str());
            ++failures;
            continue;
        }
        if (rank == 0) {
            std::printf("test_halo: np=%d %-12s (%dx%d) max guard-cell error = %.3e  %s\n", nprocs,
                        c.name, c.px, c.py, e, (e <= tol) ? "ok" : "FAIL");
            std::fflush(stdout);
        }
        if (e > tol) ++failures;
    }

    int globalFailures = 0;
    MPI_Allreduce(&failures, &globalFailures, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("test_halo: %s (%d failing case(s), tol %.1e)\n",
                    globalFailures == 0 ? "PASS" : "FAIL", globalFailures, tol);
        std::fflush(stdout);
    }

    MPI_Finalize();
    return globalFailures == 0 ? 0 : 1;
}
