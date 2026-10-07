// ranks: 1 2 4
//
// test_weno_order -- Spec "Correctness bar": advect a smooth Gaussian in a
// uniform velocity field at 32/64/128; observed order >= 4.5.
//
// What is measured
// ----------------
// The WENO5 kernel is a *spatial* operator; the Spec pairs it with forward
// Euler, which is first order in time, so a time-integrated error would
// measure dt, not the stencil (reaching the 5th-order floor at 128^2 would
// need dt ~ 1e-11).  The order reported here is therefore the order of the
// semi-discrete residual: with a uniform, divergence-free velocity (u0, v0)
// the exact value of the flux-difference operator is
//
//     rhs_exact = -( u0 dphi/dx + v0 dphi/dy )
//
// and `advectWeno` must approach it at 5th order as dx -> 0.  Both signs of
// the face velocity are exercised: u0 > 0 takes the positive branch (linear
// weights 1/10, 6/10, 3/10) and v0 < 0 the negative branch (3/10, 6/10, 1/10).
//
// The field is the Gaussian
//     phi(x,y) = exp( -((x-0.5)^2 + (y-0.5)^2) / (2 sigma^2) ),  sigma = 0.1
// whose derivatives are known in closed form.  The error norms are taken over
// a *fixed physical* window [0.15, 0.85]^2 (not a fixed number of cells), so
// the same region is compared at every resolution; the window keeps the
// zero-gradient wall extrapolation -- which is not the exact Gaussian -- out
// of the measurement, while still covering 3.5 sigma of the bump, where the
// field is O(1e-3 .. 1) and its sixth derivative is large.
//
// A silent no-op would also be "smooth": the test additionally requires that
// max|rhs| be O(1) (the exact value is ~4 for this bump), that a forward Euler
// step actually move the field, and that the error itself be small in absolute
// terms, not merely self-consistent across resolutions.

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "advect.h"
#include "grid.h"
#include "params.h"

namespace {

constexpr double kU0 = 1.0;    // positive branch in x
constexpr double kV0 = -0.5;   // negative branch in y
constexpr double kCx = 0.5;
constexpr double kCy = 0.5;
constexpr double kSigma = 0.1;
constexpr double kWinLo = 0.15;  // error window, physical coordinates
constexpr double kWinHi = 0.85;

double gaussian(double x, double y) {
    const double r2 = (x - kCx) * (x - kCx) + (y - kCy) * (y - kCy);
    return std::exp(-r2 / (2.0 * kSigma * kSigma));
}

// -(u0 dphi/dx + v0 dphi/dy) with dphi/dx = -(x-cx)/sigma^2 * phi
double exactRhs(double x, double y) {
    const double phi = gaussian(x, y);
    const double dpdx = -(x - kCx) / (kSigma * kSigma) * phi;
    const double dpdy = -(y - kCy) / (kSigma * kSigma) * phi;
    return -(kU0 * dpdx + kV0 * dpdy);
}

struct Case {
    double l1 = 0.0;
    double linf = 0.0;
    double rhsMax = 0.0;
    double stepMove = 0.0;
    bool ok = true;
};

Case runCase(int n, std::string& err) {
    Case c;

    Params p;
    p.nx = n;
    p.ny = n;

    Grid g;
    if (!g.create(p, MPI_COMM_WORLD, err)) {
        c.ok = false;
        return c;
    }

    std::vector<double> phi = g.allocate();
    std::vector<double> u = g.allocate(kU0);  // uniform everywhere, guards included
    std::vector<double> v = g.allocate(kV0);
    std::vector<double> rhs = g.allocate();

    for (int j = 0; j < g.ny(); ++j)
        for (int i = 0; i < g.nx(); ++i)
            phi[g.index(i, j)] = gaussian(g.xCenter(i), g.yCenter(j));
    g.exchange(phi);

    advectWeno(g, phi, u, v, rhs);

    // Error over the fixed physical window; a forward Euler step with a CFL
    // 0.5 timestep gives the "did anything move" check.
    const double dt = 0.5 / (std::fabs(kU0) / g.dx() + std::fabs(kV0) / g.dy());
    double l1 = 0.0;
    double linf = 0.0;
    double rhsMax = 0.0;
    double move = 0.0;
    long count = 0;
    for (int j = 0; j < g.ny(); ++j) {
        for (int i = 0; i < g.nx(); ++i) {
            const double x = g.xCenter(i);
            const double y = g.yCenter(j);
            const double r = rhs[g.index(i, j)];
            if (std::fabs(r) > rhsMax) rhsMax = std::fabs(r);
            const double d = std::fabs(dt * r);
            if (d > move) move = d;
            if (x < kWinLo || x > kWinHi || y < kWinLo || y > kWinHi) continue;
            const double e = std::fabs(r - exactRhs(x, y));
            l1 += e;
            ++count;
            if (e > linf) linf = e;
        }
    }

    double gl1 = 0.0;
    double glinf = 0.0;
    double grhs = 0.0;
    double gmove = 0.0;
    long gcount = 0;
    MPI_Allreduce(&l1, &gl1, 1, MPI_DOUBLE, MPI_SUM, g.comm());
    MPI_Allreduce(&linf, &glinf, 1, MPI_DOUBLE, MPI_MAX, g.comm());
    MPI_Allreduce(&rhsMax, &grhs, 1, MPI_DOUBLE, MPI_MAX, g.comm());
    MPI_Allreduce(&move, &gmove, 1, MPI_DOUBLE, MPI_MAX, g.comm());
    MPI_Allreduce(&count, &gcount, 1, MPI_LONG, MPI_SUM, g.comm());

    c.l1 = (gcount > 0) ? gl1 / static_cast<double>(gcount) : 0.0;
    c.linf = glinf;
    c.rhsMax = grhs;
    c.stepMove = gmove;

    g.destroy();
    return c;
}

double order(double eCoarse, double eFine) {
    if (eFine <= 0.0 || eCoarse <= 0.0) return 0.0;
    return std::log(eCoarse / eFine) / std::log(2.0);
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nprocs = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int res[3] = {32, 64, 128};
    Case c[3];
    bool ok = true;
    std::string err;

    for (int k = 0; k < 3; ++k) {
        c[k] = runCase(res[k], err);
        if (!c[k].ok) {
            if (rank == 0) std::fprintf(stderr, "test_weno_order: grid error: %s\n", err.c_str());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (rank == 0) {
            std::printf(
                "test_weno_order: np=%d %3dx%-3d  L1 err %.6e  Linf err %.6e  max|rhs| %.4f  "
                "max|dt*rhs| %.3e\n",
                nprocs, res[k], res[k], c[k].l1, c[k].linf, c[k].rhsMax, c[k].stepMove);
        }
        // Not a no-op: the exact operator peaks near 4 for this bump, and a
        // single Euler step must move the field measurably.
        if (c[k].rhsMax < 1.0) {
            ok = false;
            if (rank == 0)
                std::fprintf(stderr, "test_weno_order: FAIL %d^2: max|rhs| = %g is not O(1)\n",
                             res[k], c[k].rhsMax);
        }
        if (c[k].stepMove < 1.0e-3) {
            ok = false;
            if (rank == 0)
                std::fprintf(stderr, "test_weno_order: FAIL %d^2: Euler step moves phi by %g\n",
                             res[k], c[k].stepMove);
        }
    }

    const double o1l1 = order(c[0].l1, c[1].l1);
    const double o2l1 = order(c[1].l1, c[2].l1);
    const double o1li = order(c[0].linf, c[1].linf);
    const double o2li = order(c[1].linf, c[2].linf);

    if (rank == 0) {
        std::printf("test_weno_order: order (L1)   32->64: %.2f   64->128: %.2f\n", o1l1, o2l1);
        std::printf("test_weno_order: order (Linf) 32->64: %.2f   64->128: %.2f\n", o1li, o2li);
    }

    // Spec bar: observed order >= 4.5.  Both norms are reported; the assertion
    // is on both pairs, so a kernel that is accurate only on average cannot
    // pass.
    const double bar = 4.5;
    const double orders[4] = {o1l1, o2l1, o1li, o2li};
    const char* names[4] = {"L1 32->64", "L1 64->128", "Linf 32->64", "Linf 64->128"};
    for (int k = 0; k < 4; ++k) {
        if (!(orders[k] >= bar)) {
            ok = false;
            if (rank == 0)
                std::fprintf(stderr, "test_weno_order: FAIL %s order %.3f < %.2f\n", names[k],
                             orders[k], bar);
        }
    }

    // Absolute sanity: the finest-grid residual must actually be small.
    if (!(c[2].linf < 1.0e-3)) {
        ok = false;
        if (rank == 0)
            std::fprintf(stderr, "test_weno_order: FAIL 128^2 Linf err %.3e is not small\n",
                         c[2].linf);
    }

    if (rank == 0) {
        if (ok)
            std::printf("test_weno_order: PASS (order bar %.2f, both norms)\n", bar);
        else
            std::printf("test_weno_order: FAIL\n");
        std::fflush(stdout);
    }

    MPI_Finalize();
    return ok ? 0 : 1;
}
