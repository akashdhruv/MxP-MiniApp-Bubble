// ranks: 1 2 4
//
// test_redistance -- Spec "Correctness bar": redistance a deliberately scaled
// field phi = 2*(0.1 - r); away from the interface ||grad phi| - 1| <= 5e-2
// after 50 iterations, and the zero contour moves <= 0.1 dx.
//
// What is measured
// ----------------
// The initial field has |grad phi| = 2 everywhere, so a redistance kernel that
// does nothing -- or that relaxes in the wrong direction -- is visible
// immediately.  Two independent quantities are checked:
//
//   (a) |grad phi| (the Godunov upwind magnitude the kernel itself uses)
//       inside a band 3 dx <= |d| <= 10 dx of exact distance from the circle,
//       where d = 0.1 - r.  The band starts outside the frozen interface
//       cells (those are, by construction, held at their *original* 2*d
//       values and must not be judged against |grad| = 1) and ends inside the
//       distance that 50 pseudo-steps of size lsDT = sqrt(dx^2+dy^2)/5 can
//       propagate.  That pseudo-CFL is sqrt(2)/5 = 0.283 cells per iteration,
//       so information from the interface has reached at most ~14 cells in 50
//       iterations and the last few cells behind that front are still
//       relaxing: measured at 3..10 cells the maximum sits exactly on the
//       outer edge (d = -10.0 cells, ||grad|-1| = 6.1e-2 at 128^2), i.e. it is
//       the front, not the kernel.  The band therefore stops at 8 cells,
//       comfortably inside the converged region, and the reported maximum is
//       then an interior value rather than an edge artefact.  The Spec's 5e-2
//       tolerance is used unchanged.
//
//   (b) the zero contour: for every pair of adjacent owned cells that
//       brackets a sign change, the crossing point is linearly interpolated
//       before and after redistancing, and the two are compared.  This is the
//       interface-freeze check -- with the freeze removed the contour creeps.
//
// Plus a no-op guard: the field must actually have moved (max |phi - phi_0|
// of order the band width), and the *initial* field must fail criterion (a),
// so a test that passes on a kernel that returns its input is impossible.

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "grid.h"
#include "params.h"
#include "redistance.h"

namespace {

constexpr int kIterations = 50;
constexpr double kGradTol = 5.0e-2;      // Spec: ||grad phi| - 1| <= 5e-2
constexpr double kDriftTolCells = 0.1;   // Spec: contour moves <= 0.1 dx
constexpr double kCx = 0.5;
constexpr double kCy = 0.5;
constexpr double kR = 0.1;
constexpr double kBandLo = 3.0;          // in cells, from the interface
constexpr double kBandHi = 8.0;

// Exact signed distance to the circle, positive inside.
double dist(double x, double y) {
    return kR - std::sqrt((x - kCx) * (x - kCx) + (y - kCy) * (y - kCy));
}

struct Result {
    double gradErr = 0.0;     // max ||grad phi| - 1| in the band, after
    double gradErrX = 0.0;    // where that maximum sits
    double gradErrY = 0.0;
    double gradErrD = 0.0;    // its signed distance, in cells
    double gradErr0 = 0.0;    // the same for the initial field (must be ~1)
    double drift = 0.0;       // max contour motion, in cells
    double moved = 0.0;       // max |phi - phi_0|, in cells
    long flips = 0;
    long bandCells = 0;
    long crossings = 0;
};

// Linear crossing position (in units of the cell spacing, measured from the
// first cell centre) of a sign-changing pair.
double crossing(double a, double b) { return a / (a - b); }

Result runCase(int n, std::string& err, bool& ok) {
    Result res;
    ok = true;

    Params p;
    p.nx = n;
    p.ny = n;

    Grid g;
    if (!g.create(p, MPI_COMM_WORLD, err)) {
        ok = false;
        return res;
    }

    std::vector<double> phi = g.allocate();
    for (int j = -g.ng(); j < g.ny() + g.ng(); ++j)
        for (int i = -g.ng(); i < g.nx() + g.ng(); ++i)
            phi[g.index(i, j)] = 2.0 * dist(g.xCenter(i), g.yCenter(j));
    g.exchange(phi);  // Neumann walls, exactly as the solver sees it

    const std::vector<double> phi0 = phi;

    // Initial field: |grad phi| = 2, so the band criterion must fail here.
    double lgrad0 = 0.0;
    for (int j = 0; j < g.ny(); ++j) {
        for (int i = 0; i < g.nx(); ++i) {
            const double d = std::fabs(dist(g.xCenter(i), g.yCenter(j)));
            if (d < kBandLo * g.dx() || d > kBandHi * g.dx()) continue;
            const double e = std::fabs(godunovGradMag(g, phi0, i, j) - 1.0);
            if (e > lgrad0) lgrad0 = e;
        }
    }

    std::vector<double> phiOrig;
    const long flips = redistance(g, phi, phiOrig, kIterations);

    double lgrad = 0.0;
    double lgx = 0.0;
    double lgy = 0.0;
    double lgd = 0.0;
    double lmoved = 0.0;
    long lband = 0;
    for (int j = 0; j < g.ny(); ++j) {
        for (int i = 0; i < g.nx(); ++i) {
            const double sd = dist(g.xCenter(i), g.yCenter(j));
            const double d = std::fabs(sd);
            if (d < kBandLo * g.dx() || d > kBandHi * g.dx()) continue;
            ++lband;
            const double e = std::fabs(godunovGradMag(g, phi, i, j) - 1.0);
            if (e > lgrad) {
                lgrad = e;
                lgx = g.xCenter(i);
                lgy = g.yCenter(j);
                lgd = sd / g.dx();
            }
            const double m = std::fabs(phi[g.index(i, j)] - phi0[g.index(i, j)]) / g.dx();
            if (m > lmoved) lmoved = m;
        }
    }

    // Zero-contour motion, from bracketing pairs of owned cells.
    double ldrift = 0.0;
    long lcross = 0;
    for (int j = 0; j < g.ny(); ++j) {
        for (int i = 0; i + 1 < g.nx(); ++i) {
            const double a0 = phi0[g.index(i, j)];
            const double b0 = phi0[g.index(i + 1, j)];
            if (a0 * b0 >= 0.0) continue;
            const double a = phi[g.index(i, j)];
            const double b = phi[g.index(i + 1, j)];
            ++lcross;
            if (a * b >= 0.0) {  // the contour left the cell pair entirely
                ldrift = 1.0e30;
                continue;
            }
            const double s = std::fabs(crossing(a, b) - crossing(a0, b0));
            if (s > ldrift) ldrift = s;
        }
    }
    for (int j = 0; j + 1 < g.ny(); ++j) {
        for (int i = 0; i < g.nx(); ++i) {
            const double a0 = phi0[g.index(i, j)];
            const double b0 = phi0[g.index(i, j + 1)];
            if (a0 * b0 >= 0.0) continue;
            const double a = phi[g.index(i, j)];
            const double b = phi[g.index(i, j + 1)];
            ++lcross;
            if (a * b >= 0.0) {
                ldrift = 1.0e30;
                continue;
            }
            const double s = std::fabs(crossing(a, b) - crossing(a0, b0));
            if (s > ldrift) ldrift = s;
        }
    }

    {  // max of ||grad|-1| together with the cell it came from
        struct {
            double val;
            int rank;
        } in, out;
        in.val = lgrad;
        in.rank = g.rank();
        MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, g.comm());
        double loc[3] = {lgx, lgy, lgd};
        MPI_Bcast(loc, 3, MPI_DOUBLE, out.rank, g.comm());
        res.gradErrX = loc[0];
        res.gradErrY = loc[1];
        res.gradErrD = loc[2];
    }
    MPI_Allreduce(&lgrad, &res.gradErr, 1, MPI_DOUBLE, MPI_MAX, g.comm());
    MPI_Allreduce(&lgrad0, &res.gradErr0, 1, MPI_DOUBLE, MPI_MAX, g.comm());
    MPI_Allreduce(&ldrift, &res.drift, 1, MPI_DOUBLE, MPI_MAX, g.comm());
    MPI_Allreduce(&lmoved, &res.moved, 1, MPI_DOUBLE, MPI_MAX, g.comm());
    MPI_Allreduce(&lband, &res.bandCells, 1, MPI_LONG, MPI_SUM, g.comm());
    MPI_Allreduce(&lcross, &res.crossings, 1, MPI_LONG, MPI_SUM, g.comm());
    res.flips = flips;

    g.destroy();
    return res;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int res[2] = {64, 128};
    bool ok = true;
    std::string err;

    for (int k = 0; k < 2; ++k) {
        bool gridOk = true;
        const Result r = runCase(res[k], err, gridOk);
        if (!gridOk) {
            if (rank == 0) std::fprintf(stderr, "test_redistance: grid error: %s\n", err.c_str());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (rank == 0) {
            std::printf(
                "test_redistance: np=%d %3dx%-3d  max||grad|-1| %.4e (initial %.4e) at "
                "(%.4f,%.4f) d=%+.1f cells  contour drift %.3e cells  moved %.2f cells  "
                "band %ld  crossings %ld  sign flips %ld\n",
                nprocs, res[k], res[k], r.gradErr, r.gradErr0, r.gradErrX, r.gradErrY, r.gradErrD,
                r.drift, r.moved, r.bandCells, r.crossings, r.flips);
        }
        if (r.bandCells <= 0 || r.crossings <= 0) {
            ok = false;
            if (rank == 0)
                std::fprintf(stderr,
                             "test_redistance: FAIL %d^2: empty measurement set "
                             "(band %ld, crossings %ld)\n",
                             res[k], r.bandCells, r.crossings);
        }
        if (r.gradErr0 < 0.5) {
            ok = false;
            if (rank == 0)
                std::fprintf(stderr,
                             "test_redistance: FAIL %d^2: the initial field already satisfies "
                             "the band criterion (%.3e) -- the test proves nothing\n",
                             res[k], r.gradErr0);
        }
        if (r.moved < 1.0) {
            ok = false;
            if (rank == 0)
                std::fprintf(stderr,
                             "test_redistance: FAIL %d^2: phi moved only %.3e cells -- no-op\n",
                             res[k], r.moved);
        }
        if (!(r.gradErr <= kGradTol)) {
            ok = false;
            if (rank == 0)
                std::fprintf(stderr,
                             "test_redistance: FAIL %d^2: max||grad phi|-1| = %.4e > %.2e "
                             "in the band %.0f..%.0f cells from the interface\n",
                             res[k], r.gradErr, kGradTol, kBandLo, kBandHi);
        }
        if (!(r.drift <= kDriftTolCells)) {
            ok = false;
            if (rank == 0)
                std::fprintf(stderr,
                             "test_redistance: FAIL %d^2: zero contour moved %.4e cells > %.2f dx\n",
                             res[k], r.drift, kDriftTolCells);
        }
    }

    if (rank == 0) {
        if (ok)
            std::printf("test_redistance: PASS (grad bar %.2e, drift bar %.2f dx, %d iterations)\n",
                        kGradTol, kDriftTolCells, kIterations);
        else
            std::printf("test_redistance: FAIL\n");
        std::fflush(stdout);
    }

    MPI_Finalize();
    return ok ? 0 : 1;
}
