// ranks: 1 2 4
//
// test_divergence -- Spec "Correctness bar": invariant 6 on the initialized
// field, 128^2 and 64^2.
//
//   max |(u(i+1,j)-u(i,j))/dx + (v(i,j+1)-v(i,j))/dy| <= 1e-12
//
// A silent no-op (all-zero velocity, untouched level set) has zero divergence
// too, so the test also demands that the field actually be the DeformingBubble
// field:
//   * the level set equals the analytic circle (evaluated here from the global
//     cell index, independently of Grid's local offsets) to roundoff, in the
//     guard cells as well as the owned cells, and is not identically zero;
//   * max|u| and max|v| are O(1) -- the continuum maxima of
//     sin^2(pi x) sin(2 pi y) are 1 -- so an all-zero field fails;
//   * the normal velocity vanishes on the four physical walls, so nothing
//     advects out of the domain.

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "grid.h"
#include "init.h"
#include "params.h"

namespace {

constexpr double kSentinel = -1.0e30;

int clampIndex(int i, int n) {
    if (i < 0) return 0;
    if (i > n - 1) return n - 1;
    return i;
}

struct Result {
    double div = 0.0;
    double phiErr = 0.0;
    double umax = 0.0;
    double vmax = 0.0;
    double wallVel = 0.0;
    double phiAbsMax = 0.0;
    bool ok = true;
};

Result runCase(int nx, int ny, bool reverseTime, double time, std::string& err) {
    Result r;

    Params p;
    p.nx = nx;
    p.ny = ny;
    p.reverse_time = reverseTime;
    p.tmax = 2.0;

    Grid g;
    if (!g.create(p, MPI_COMM_WORLD, err)) {
        r.ok = false;
        return r;
    }

    std::vector<double> dfun = g.allocate(kSentinel);
    std::vector<double> u = g.allocate(kSentinel);
    std::vector<double> v = g.allocate(kSentinel);

    initLevelSet(g, p, dfun);
    computeFaceVelocity(g, p, time, u, v);

    r.div = maxDivergence(g, u, v);
    maxFaceSpeeds(g, u, v, r.umax, r.vmax);

    // Level set versus the analytic circle, owned cells and guard cells.
    // Guard-cell expectation: the global index, clamped at a physical wall
    // (zero-gradient) and unclamped across a rank boundary.
    double phiErr = 0.0;
    double phiAbsMax = 0.0;
    int unfilled = 0;
    for (int j = -g.ng(); j < g.ny() + g.ng(); ++j) {
        for (int i = -g.ng(); i < g.nx() + g.ng(); ++i) {
            const double got = dfun[g.index(i, j)];
            if (got == kSentinel) {
                ++unfilled;
                continue;
            }
            const int gi = clampIndex(g.i0() + i, g.nxGlobal());
            const int gj = clampIndex(g.j0() + j, g.nyGlobal());
            const double x = p.xmin + (static_cast<double>(gi) + 0.5) * g.dx();
            const double y = p.ymin + (static_cast<double>(gj) + 0.5) * g.dy();
            const double want = levelSetAnalytic(p, x, y);
            const double e = std::fabs(got - want);
            if (e > phiErr) phiErr = e;
            if (std::fabs(got) > phiAbsMax) phiAbsMax = std::fabs(got);
        }
    }
    if (unfilled > 0) {
        err = "rank " + std::to_string(g.rank()) + ": " + std::to_string(unfilled) +
              " level-set cells were never filled";
        r.ok = false;
        g.destroy();
        return r;
    }

    // Normal velocity on the physical walls.
    double wall = 0.0;
    if (g.atWallXlo()) {
        for (int j = 0; j < g.ny(); ++j) wall = std::fmax(wall, std::fabs(u[g.index(0, j)]));
    }
    if (g.atWallXhi()) {
        for (int j = 0; j < g.ny(); ++j) wall = std::fmax(wall, std::fabs(u[g.index(g.nx(), j)]));
    }
    if (g.atWallYlo()) {
        for (int i = 0; i < g.nx(); ++i) wall = std::fmax(wall, std::fabs(v[g.index(i, 0)]));
    }
    if (g.atWallYhi()) {
        for (int i = 0; i < g.nx(); ++i) wall = std::fmax(wall, std::fabs(v[g.index(i, g.ny())]));
    }

    double in[3] = {phiErr, phiAbsMax, wall};
    double out[3] = {0.0, 0.0, 0.0};
    MPI_Allreduce(in, out, 3, MPI_DOUBLE, MPI_MAX, g.comm());
    r.phiErr = out[0];
    r.phiAbsMax = out[1];
    r.wallVel = out[2];

    g.destroy();
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const double divTol = 1.0e-12;   // invariant 6
    const double phiTol = 1.0e-15;   // "matches the analytic circle to roundoff"
    const double wallTol = 1.0e-15;  // the prescribed field vanishes on the walls

    struct Case {
        const char* name;
        int nx;
        int ny;
        bool reverse;
        double t;
    };
    const Case cases[3] = {{"128x128", 128, 128, false, 0.0},
                           {"64x64", 64, 64, false, 0.0},
                           {"64x64 reversed t=0.3", 64, 64, true, 0.3}};

    int failures = 0;
    for (const Case& c : cases) {
        std::string err;
        const Result r = runCase(c.nx, c.ny, c.reverse, c.t, err);
        if (!r.ok) {
            if (rank == 0) std::printf("test_divergence: %s SETUP FAIL: %s\n", c.name, err.c_str());
            ++failures;
            continue;
        }
        // The expected scale of the field; with reverse_time the amplitude is
        // |cos(pi t / tmax)| of the steady field.
        const double amp = c.reverse ? std::fabs(std::cos(3.14159265358979323846 * c.t / 2.0)) : 1.0;
        const bool ok = (r.div <= divTol) && (r.phiErr <= phiTol) && (r.wallVel <= wallTol) &&
                        (r.umax > 0.5 * amp) && (r.umax < 1.5 * amp) && (r.vmax > 0.5 * amp) &&
                        (r.vmax < 1.5 * amp) && (r.phiAbsMax > 0.1);
        if (!ok) ++failures;
        if (rank == 0) {
            std::printf(
                "test_divergence: np=%d %-21s max|div| = %.3e  phi err = %.3e  "
                "max|u| = %.6f  max|v| = %.6f  wall |u.n| = %.3e  %s\n",
                nprocs, c.name, r.div, r.phiErr, r.umax, r.vmax, r.wallVel, ok ? "ok" : "FAIL");
            std::fflush(stdout);
        }
    }

    int globalFailures = 0;
    MPI_Allreduce(&failures, &globalFailures, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("test_divergence: %s (%d failing case(s), div tol %.1e, phi tol %.1e)\n",
                    globalFailures == 0 ? "PASS" : "FAIL", globalFailures, divTol, phiTol);
        std::fflush(stdout);
    }

    MPI_Finalize();
    return globalFailures == 0 ? 0 : 1;
}
