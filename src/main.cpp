// main.cpp -- entry point for the Deforming Bubble mini-app (U7).
//
//   mpirun -np N ./bubble [params.ini] [key=value ...]
//
// MPI_Init -> parse and validate the runtime parameters -> build the Grid
// (Cartesian communicator, ownership map, guard cells) -> seed the bubble
// level set -> run the time loop in driver.cpp -> report -> clean finalize.
//
// Trailing `key=value` arguments override the file, which is how the
// validation runs (reverse_time, max_steps, outdir) are driven without
// touching params/bubble.ini.

#include <mpi.h>

#include <cstdio>
#include <string>
#include <vector>

#include "driver.h"
#include "grid.h"
#include "init.h"
#include "io.h"
#include "params.h"

namespace {

void abortWith(const std::string& msg, int rank) {
    if (rank == 0) std::fprintf(stderr, "bubble: error: %s\n", msg.c_str());
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    std::string paramFile = "params/bubble.ini";
    std::vector<std::string> overrides;
    for (int a = 1; a < argc; ++a) {
        const std::string arg(argv[a]);
        if (arg == "-h" || arg == "--help") {
            if (rank == 0) {
                std::printf("usage: mpirun -np N ./bubble [params.ini] [key=value ...]\n");
            }
            MPI_Finalize();
            return 0;
        }
        if (a == 1 && arg.find('=') == std::string::npos) {
            paramFile = arg;
        } else {
            overrides.push_back(arg);
        }
    }

    Params p;
    std::string err;
    if (!parseParamFile(paramFile, p, err)) abortWith(err, rank);
    for (std::size_t k = 0; k < overrides.size(); ++k) {
        if (!parseParamAssignment(overrides[k], p, err)) abortWith(err, rank);
    }
    if (!validateParams(p, err)) abortWith(err, rank);

    {  // Grid owns an MPI communicator: scope it so it is freed before finalize.
        Grid grid;
        if (!grid.create(p, MPI_COMM_WORLD, err)) abortWith(err, rank);

        if (grid.rank() == 0) {
            std::printf("bubble: Deforming Bubble mini-app\n");
            std::printf("bubble: params         : %s", paramFile.c_str());
            for (std::size_t k = 0; k < overrides.size(); ++k) {
                std::printf("  [%s]", overrides[k].c_str());
            }
            std::printf("\n");
            std::printf("bubble: global mesh    : %d x %d  on [%g,%g] x [%g,%g]\n", grid.nxGlobal(),
                        grid.nyGlobal(), p.xmin, p.xmax, p.ymin, p.ymax);
            std::printf("bubble: cell size      : dx %g  dy %g\n", grid.dx(), grid.dy());
            std::printf("bubble: guard cells    : %d\n", grid.ng());
            std::printf("bubble: process grid   : %d x %d  (%d ranks)\n", grid.dim(0), grid.dim(1),
                        nprocs);
            std::printf("bubble: local mesh     : %d x %d\n", grid.nx(), grid.ny());
            std::printf("bubble: tmax %g  cfl %g  fixed_dt %g  max_steps %d  reverse_time %s\n",
                        p.tmax, p.cfl, p.fixed_dt, p.max_steps, p.reverse_time ? "true" : "false");
            std::printf("bubble: ls_iterations  : %d\n", p.ls_iterations);
            std::printf("bubble: outdir         : %s  (plot every %d, history every %d)\n",
                        p.outdir.c_str(), p.plot_interval, p.hist_interval);
            std::fflush(stdout);
        }
        MPI_Barrier(grid.comm());

        if (p.verbosity > 1) {  // rank layout, one line per rank, in rank order
            for (int r = 0; r < nprocs; ++r) {
                if (r == grid.rank()) {
                    std::printf(
                        "bubble: rank %3d/%d  coords (%d,%d)  owns i[%d,%d) j[%d,%d)  "
                        "nbr x(%d,%d) y(%d,%d)\n",
                        grid.rank(), nprocs, grid.coord(0), grid.coord(1), grid.i0(),
                        grid.i0() + grid.nx(), grid.j0(), grid.j0() + grid.ny(),
                        grid.neighborXlo(), grid.neighborXhi(), grid.neighborYlo(),
                        grid.neighborYhi());
                    std::fflush(stdout);
                }
                MPI_Barrier(grid.comm());
            }
        }

        // Initial condition (U2).
        std::vector<double> dfun = grid.allocate();
        initLevelSet(grid, p, dfun);

        // Deviation from the analytic circle at t = 0 (should be exactly 0).
        double lphiErr = 0.0;
        for (int j = 0; j < grid.ny(); ++j) {
            for (int i = 0; i < grid.nx(); ++i) {
                const double want = levelSetAnalytic(p, grid.xCenter(i), grid.yCenter(j));
                const double e = dfun[grid.index(i, j)] - want;
                const double ae = (e < 0.0) ? -e : e;
                if (ae > lphiErr) lphiErr = ae;
            }
        }
        double phiErr = 0.0;
        MPI_Allreduce(&lphiErr, &phiErr, 1, MPI_DOUBLE, MPI_MAX, grid.comm());

        // Time loop (U6).
        const double t0 = MPI_Wtime();
        RunSummary s;
        if (!runSimulation(grid, p, dfun, s, err)) abortWith(err, rank);
        const double wall = MPI_Wtime() - t0;

        // Final deviation from the initial circle: the reversal-mode error
        // metric (meaningful only with reverse_time = true, reported always).
        double lrevErr = 0.0;
        for (int j = 0; j < grid.ny(); ++j) {
            for (int i = 0; i < grid.nx(); ++i) {
                const double want = levelSetAnalytic(p, grid.xCenter(i), grid.yCenter(j));
                const double e = dfun[grid.index(i, j)] - want;
                const double ae = (e < 0.0) ? -e : e;
                if (ae > lrevErr) lrevErr = ae;
            }
        }
        double revErr = 0.0;
        MPI_Allreduce(&lrevErr, &revErr, 1, MPI_DOUBLE, MPI_MAX, grid.comm());

        if (grid.rank() == 0) {
            std::printf("bubble: init dfun      : analytic err %.3e\n", phiErr);
            std::printf("bubble: steps          : %d  to t = %.6f  (last dt %.6e)\n", s.steps,
                        s.time, s.dt_last);
            std::printf("bubble: area           : %.9f -> %.9f   drift %.4f%%\n", s.area_initial,
                        s.area_final, 100.0 * s.area_drift);
            std::printf("bubble: max divergence : %.3e  (invariant 6 tol 1.0e-12)\n",
                        s.max_divergence);
            std::printf("bubble: sign flips     : %ld (total over the run)\n", s.sign_flips_total);
            std::printf("bubble: |dfun - dfun(t=0)| at t_end : %.6e%s\n", revErr,
                        p.reverse_time ? "   (reversal-mode error)" : "   (deformed, not reversed)");
            std::printf("bubble: output         : %d plot(s) + %d history row(s) in %s/\n",
                        s.plots_written, s.history_rows, p.outdir.c_str());
            std::printf("bubble: wall time      : %.2f s on %d rank(s)\n", wall, nprocs);
            std::printf("bubble: done\n");
            std::fflush(stdout);
        }

        grid.destroy();
    }

    MPI_Finalize();
    return 0;
}
