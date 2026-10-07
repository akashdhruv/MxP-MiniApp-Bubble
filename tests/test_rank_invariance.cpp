// ranks: 1 2 4
//
// test_rank_invariance -- Spec "Correctness bar": 20 steps at -np 1 vs -np 4,
// max-norm difference <= 1e-12 (invariant 4).
//
// The comparison is self-contained inside one mpirun: the field is advanced 20
// steps on the real decomposition, the owned blocks are gathered onto rank 0,
// and rank 0 then advances the *same* initial condition 20 steps on a private
// single-rank grid built over MPI_COMM_SELF.  Running the binary on 1, 2 and 4
// ranks therefore compares 1-vs-1, 2-vs-1 and 4-vs-1 decompositions of the
// identical algorithm.  Any missing halo exchange (in particular the one
// *between* the two redistance iterations) shows up here, which is why the run
// is 20 steps and not one.

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "driver.h"
#include "grid.h"
#include "init.h"
#include "params.h"

namespace {

const int kSteps = 20;
const double kTol = 1.0e-12;

Params testParams() {
    Params p;
    p.nx = 64;
    p.ny = 64;
    p.tmax = 2.0;
    p.cfl = 0.5;
    p.fixed_dt = 1.0e-3;  // identical dt on every rank count, by construction
    p.max_steps = kSteps;
    p.ls_iterations = 2;
    p.verbosity = 0;
    return p;
}

// Advance `phi` kSteps steps with the driver's per-step sequence.
void advance(const Grid& g, const Params& p, std::vector<double>& phi, long& flips) {
    std::vector<double> u = g.allocate();
    std::vector<double> v = g.allocate();
    std::vector<double> rhs = g.allocate();
    std::vector<double> phiOrig = g.allocate();
    double time = 0.0;
    flips = 0;
    for (int s = 0; s < kSteps; ++s) {
        const double dt = computeTimestep(p, g, 1.0, 1.0, time);
        computeFaceVelocity(g, p, time, u, v);
        flips += advanceOneStep(g, p, phi, u, v, dt, rhs, phiOrig);
        time += dt;
    }
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    const Params p = testParams();
    std::string err;
    int failures = 0;

    {
        Grid g;
        if (!g.create(p, MPI_COMM_WORLD, err)) {
            std::fprintf(stderr, "test_rank_invariance: grid.create failed: %s\n", err.c_str());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::vector<double> phi = g.allocate();
        initLevelSet(g, p, phi);
        long flipsPar = 0;
        advance(g, p, phi, flipsPar);

        // Gather the owned blocks onto rank 0: disjoint blocks, so a sum over
        // zero-padded global arrays is exact.
        const std::size_t n = static_cast<std::size_t>(p.nx) * static_cast<std::size_t>(p.ny);
        std::vector<double> localGlobal(n, 0.0);
        for (int j = 0; j < g.ny(); ++j) {
            for (int i = 0; i < g.nx(); ++i) {
                const std::size_t k =
                    static_cast<std::size_t>(g.j0() + j) * static_cast<std::size_t>(p.nx) +
                    static_cast<std::size_t>(g.i0() + i);
                localGlobal[k] = phi[g.index(i, j)];
            }
        }
        std::vector<double> parallelField(n, 0.0);
        MPI_Reduce(localGlobal.data(), parallelField.data(), static_cast<int>(n), MPI_DOUBLE,
                   MPI_SUM, 0, g.comm());

        int rankInComm = 0;
        MPI_Comm_rank(g.comm(), &rankInComm);

        if (rankInComm == 0) {
            // Serial reference on a private single-rank communicator.
            Grid s;
            if (!s.create(p, MPI_COMM_SELF, err)) {
                std::fprintf(stderr, "test_rank_invariance: serial grid.create failed: %s\n",
                             err.c_str());
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            std::vector<double> sphi = s.allocate();
            initLevelSet(s, p, sphi);
            const std::vector<double> sphi0 = sphi;  // for the anti-no-op guard
            long flipsSer = 0;
            advance(s, p, sphi, flipsSer);

            // Anti-no-op guard: a solver that did nothing would also agree
            // across rank counts, so require the 20 steps to have actually
            // moved the field.
            double maxMove = 0.0;
            for (int j = 0; j < p.ny; ++j) {
                for (int i = 0; i < p.nx; ++i) {
                    const double m = std::fabs(sphi[s.index(i, j)] - sphi0[s.index(i, j)]);
                    if (m > maxMove) maxMove = m;
                }
            }
            if (!(maxMove > 1.0e-6)) {
                std::printf(
                    "test_rank_invariance: FAIL: the %d steps moved dfun by only %.3e -- the "
                    "comparison would be vacuous\n",
                    kSteps, maxMove);
                ++failures;
            }

            double maxDiff = 0.0;
            int argi = -1;
            int argj = -1;
            for (int j = 0; j < p.ny; ++j) {
                for (int i = 0; i < p.nx; ++i) {
                    const std::size_t k = static_cast<std::size_t>(j) *
                                              static_cast<std::size_t>(p.nx) +
                                          static_cast<std::size_t>(i);
                    const double d = std::fabs(parallelField[k] - sphi[s.index(i, j)]);
                    if (d > maxDiff) {
                        maxDiff = d;
                        argi = i;
                        argj = j;
                    }
                }
            }
            std::printf(
                "test_rank_invariance: np=%d  %d steps  max|dfun(np=%d) - dfun(np=1)| = %.3e "
                "at (%d,%d)  flips par/ser %ld/%ld  max|dfun - dfun(t=0)| %.3e\n",
                worldSize, kSteps, worldSize, maxDiff, argi, argj, flipsPar, flipsSer, maxMove);
            if (!(maxDiff <= kTol)) {
                std::printf(
                    "test_rank_invariance: FAIL np=%d: max-norm difference %.3e > %.2e after %d "
                    "steps\n",
                    worldSize, maxDiff, kTol, kSteps);
                ++failures;
            }
            if (flipsPar != flipsSer) {
                std::printf(
                    "test_rank_invariance: FAIL np=%d: sign-flip count %ld != serial %ld\n",
                    worldSize, flipsPar, flipsSer);
                ++failures;
            }
            s.destroy();
        }
        g.destroy();
    }

    MPI_Bcast(&failures, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (worldRank == 0) {
        std::printf("test_rank_invariance: %s\n", failures == 0 ? "PASS" : "FAIL");
        std::fflush(stdout);
    }

    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
