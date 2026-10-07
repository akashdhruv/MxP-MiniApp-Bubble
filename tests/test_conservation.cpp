// ranks: 1 4
//
// test_conservation -- Spec, "Correctness bar":
//     test_conservation | full run to tmax: bubble area drift <= 5%
//
// "Area" is the Spec's definition (Spec, "Output and post-processing"):
//     area = sum(H(phi)) dx dy,  H a smoothed Heaviside over 1.5 cells
// which is exactly what src/io.cpp writes into out/history.csv, so this test
// measures the same quantity the run reports.  The 5% bar is the Spec's and is
// NOT adjustable here (Plan, "Escalation": a tolerance that looks wrong is a
// blocked unit and a question for the human, never a loosened constant).
//
// The test drives the real kernels through driver.h's computeTimestep() and
// advanceOneStep() rather than runSimulation(), for one reason only: it must
// not truncate out/history.csv or overwrite the .vti/.pvti set that `make
// check` runs alongside.  Every numerical component (velocity, WENO5, Euler,
// redistance, diagnostics) is the production one.
//
// Alongside the Spec's measure the test also reports a *sharp* area, the plain
// count of cells with phi > 0 times dx*dy.  The two agree for a
// well-resolved interface and separate once the filament is thinner than the
// smoothing width, which is the diagnostic U9 needs.

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "driver.h"
#include "grid.h"
#include "init.h"
#include "io.h"
#include "params.h"
#include "redistance.h"

namespace {

struct Areas {
    double smoothed = 0.0;  // Spec: sum H(phi) dx dy, H over 1.5 cells
    double sharp = 0.0;     // sum [phi > 0] dx dy
};

Areas measureAreas(const Grid& g, const std::vector<double>& phi) {
    const double eps = heavisideWidth(g);
    const double cellArea = g.dx() * g.dy();
    double local[2] = {0.0, 0.0};
    for (int j = 0; j < g.ny(); ++j) {
        for (int i = 0; i < g.nx(); ++i) {
            const double f = phi[g.index(i, j)];
            local[0] += smoothedHeaviside(f, eps);
            local[1] += (f > 0.0) ? 1.0 : 0.0;
        }
    }
    double global[2] = {0.0, 0.0};
    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, g.comm());
    Areas a;
    a.smoothed = global[0] * cellArea;
    a.sharp = global[1] * cellArea;
    return a;
}

// max |a - b| over owned cells, reduced over the communicator.
double maxNormDiff(const Grid& g, const std::vector<double>& a, const std::vector<double>& b) {
    double local = 0.0;
    for (int j = 0; j < g.ny(); ++j) {
        for (int i = 0; i < g.nx(); ++i) {
            const double d = std::fabs(a[g.index(i, j)] - b[g.index(i, j)]);
            if (d > local) local = d;
        }
    }
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, g.comm());
    return global;
}

struct RunResult {
    Areas a0;
    Areas a1;
    double drift_smoothed = 0.0;
    double drift_sharp = 0.0;
    double max_divergence = 0.0;
    double phi_min = 0.0;
    double phi_max = 0.0;
    double max_norm_vs_initial = 0.0;
    double perimeter = 0.0;  // sum of cell faces across which phi changes sign, x h
    int steps = 0;
    double time = 0.0;
    long sign_flips = 0;
};

// Full run to p.tmax with the production kernels and no file output.
RunResult runToTmax(const Grid& g, const Params& p) {
    std::vector<double> phi;
    initLevelSet(g, p, phi);
    const std::vector<double> phi0 = phi;

    std::vector<double> u = g.allocate();
    std::vector<double> v = g.allocate();
    std::vector<double> rhs = g.allocate();
    std::vector<double> phiOrig = g.allocate();

    // Unscaled speeds for the CFL estimate, exactly as runSimulation does.
    {
        Params q = p;
        q.reverse_time = false;
        computeFaceVelocity(g, q, 0.0, u, v);
    }
    double umax = 0.0;
    double vmax = 0.0;
    maxFaceSpeeds(g, u, v, umax, vmax);

    RunResult r;
    r.a0 = measureAreas(g, phi);

    double time = 0.0;
    int step = 0;
    const double tEps = 1.0e-12 * p.tmax;
    while (time < p.tmax - tEps) {
        if (p.max_steps > 0 && step >= p.max_steps) break;
        const double dt = computeTimestep(p, g, umax, vmax, time);
        if (dt <= 0.0) break;
        computeFaceVelocity(g, p, time, u, v);
        const double div = maxDivergence(g, u, v);
        if (div > r.max_divergence) r.max_divergence = div;
        r.sign_flips += advanceOneStep(g, p, phi, u, v, dt, rhs, phiOrig);
        time += dt;
        ++step;
    }

    r.a1 = measureAreas(g, phi);
    r.drift_smoothed =
        (r.a0.smoothed > 0.0) ? std::fabs(r.a1.smoothed - r.a0.smoothed) / r.a0.smoothed : 0.0;
    r.drift_sharp = (r.a0.sharp > 0.0) ? std::fabs(r.a1.sharp - r.a0.sharp) / r.a0.sharp : 0.0;
    r.max_norm_vs_initial = maxNormDiff(g, phi, phi0);

    // Interface length estimate: every face whose two cells differ in sign
    // contributes one cell edge.  A closed filament of centreline length L has
    // perimeter ~ 2 L, so the thinnest structure a sign field can represent
    // covers at least L*h = perimeter*h/2 of area -- the sub-grid floor.
    {
        double lp = 0.0;
        for (int j = 0; j < g.ny(); ++j) {
            for (int i = 0; i < g.nx(); ++i) {
                const double c = phi[g.index(i, j)];
                if (c * phi[g.index(i + 1, j)] < 0.0) lp += g.dy();
                if (c * phi[g.index(i, j + 1)] < 0.0) lp += g.dx();
            }
        }
        MPI_Allreduce(&lp, &r.perimeter, 1, MPI_DOUBLE, MPI_SUM, g.comm());
    }

    r.steps = step;
    r.time = time;

    const Diagnostics d = computeDiagnostics(g, phi, u, v, 0L);
    r.phi_min = d.phi_min;
    r.phi_max = d.phi_max;
    return r;
}

// --------------------------------------------------------------------------
// Diagnostic sweep (argv flag `--sweep`, never run by `make check`).
// Question it answers: is the baseline area drift a discretisation error of an
// under-resolved filament (it then falls with h and is insensitive to dt) or a
// kernel bug (dt- and h-independent)?  ls_iterations = 0 isolates redistance.
void sweepCase(int rank, const char* label, int nx, double cfl, int lsIt, double tmax = 2.0) {
    Params p;
    p.nx = nx;
    p.ny = nx;
    p.tmax = tmax;
    p.cfl = cfl;
    p.ls_iterations = lsIt;
    p.verbosity = 0;

    Grid g;
    std::string err;
    if (!g.create(p, MPI_COMM_WORLD, err)) {
        if (rank == 0) std::printf("sweep: grid failed: %s\n", err.c_str());
        return;
    }
    const RunResult r = runToTmax(g, p);
    if (rank == 0) {
        std::printf("sweep %-22s t %4.2f nx %4d cfl %6.3f lsIt %d | steps %5d | area %.9g ->"
                    " %.9g  drift %8.3f%%  (sharp %8.3f%%)  perim %7.4f  floor(P h/2) %.6f\n",
                    label, tmax, nx, cfl, lsIt, r.steps, r.a0.smoothed, r.a1.smoothed,
                    100.0 * r.drift_smoothed, 100.0 * r.drift_sharp, r.perimeter,
                    0.5 * r.perimeter * g.dx());
        std::fflush(stdout);
    }
    g.destroy();
}

void runSweep(int rank) {
    if (rank == 0) std::printf("--- resolution sweep (cfl 0.5, ls_iterations 2) ---\n");
    sweepCase(rank, "resolution", 64, 0.5, 2);
    sweepCase(rank, "resolution", 128, 0.5, 2);
    sweepCase(rank, "resolution", 192, 0.5, 2);
    sweepCase(rank, "resolution", 256, 0.5, 2);
    if (rank == 0) std::printf("--- cfl sweep (128^2, ls_iterations 2) ---\n");
    sweepCase(rank, "cfl", 128, 0.25, 2);
    sweepCase(rank, "cfl", 128, 0.10, 2);
    sweepCase(rank, "cfl", 128, 0.02, 2);
    if (rank == 0) std::printf("--- redistance on/off (cfl 0.5) ---\n");
    sweepCase(rank, "no-redistance", 128, 0.5, 0);
    sweepCase(rank, "no-redistance", 256, 0.5, 0);
    sweepCase(rank, "no-redistance-lowcfl", 128, 0.1, 0);
}

// --------------------------------------------------------------------------
// Probe (argv flag `--probe`): redistance with NO advection at all.  The Spec
// says the interface-cell freeze is what keeps the zero contour still, so
// relaxing an exact signed-distance circle must be a (near) no-op: area must
// not move and phi must stay within O(h^2) of 0.1 - r.  Any drift here is a
// redistance bug, with advection removed from the picture entirely.
void runProbe(int rank) {
    for (int nx : {64, 128, 256}) {
        Params p;
        p.nx = nx;
        p.ny = nx;
        p.ls_iterations = 2;
        p.verbosity = 0;

        Grid g;
        std::string err;
        if (!g.create(p, MPI_COMM_WORLD, err)) {
            if (rank == 0) std::printf("probe: grid failed: %s\n", err.c_str());
            return;
        }
        std::vector<double> phi;
        initLevelSet(g, p, phi);
        std::vector<double> phiOrig = g.allocate();
        const Areas a0 = measureAreas(g, phi);

        long flips = 0;
        const int calls = 1024;
        for (int k = 0; k < calls; ++k) flips += redistance(g, phi, phiOrig, p.ls_iterations);

        const Areas a1 = measureAreas(g, phi);

        // Departure from the analytic circle, restricted to |phi| < 5h where
        // the signed distance of a circle is what redistance should preserve.
        double lerr = 0.0;
        for (int j = 0; j < g.ny(); ++j) {
            for (int i = 0; i < g.nx(); ++i) {
                const double x = g.xCenter(i);
                const double y = g.yCenter(j);
                const double ex = p.bubble_r - std::sqrt((x - p.bubble_x) * (x - p.bubble_x) +
                                                         (y - p.bubble_y) * (y - p.bubble_y));
                if (std::fabs(ex) > 5.0 * g.dx()) continue;
                const double d = std::fabs(phi[g.index(i, j)] - ex);
                if (d > lerr) lerr = d;
            }
        }
        double gerr = 0.0;
        MPI_Allreduce(&lerr, &gerr, 1, MPI_DOUBLE, MPI_MAX, g.comm());

        if (rank == 0) {
            std::printf("probe static-redistance nx %4d | %d calls x 2 iters | area %.9g -> %.9g"
                        "  drift %7.3f%% (sharp %7.3f%%)  max|phi-exact| %.3e (= %.3f h)"
                        "  flips %ld\n",
                        nx, calls, a0.smoothed, a1.smoothed,
                        100.0 * std::fabs(a1.smoothed - a0.smoothed) / a0.smoothed,
                        100.0 * std::fabs(a1.sharp - a0.sharp) / a0.sharp, gerr, gerr / g.dx(),
                        flips);
            std::fflush(stdout);
        }
        g.destroy();
    }

    // How the full-run drift scales with the number of redistance iterations
    // per step, at a cfl where advection alone is clean (1.3%).
    if (rank == 0) std::printf("--- ls_iterations sweep (128^2, cfl 0.1) ---\n");
    for (int it : {0, 1, 2, 4}) sweepCase(rank, "ls_iterations", 128, 0.1, it);

    // Convergence while the interface is still resolved: stop the same run at
    // t = 0.5 and t = 1.0 and refine.  If the kernels are faithful the drift
    // must fall with h here, and only stop falling once the filament goes
    // sub-grid (which is what t = 2 is).
    if (rank == 0) std::printf("--- early-time convergence (cfl 0.25, ls_iterations 2) ---\n");
    for (double tm : {0.5, 1.0}) {
        for (int nx : {64, 128, 256}) sweepCase(rank, "converge", nx, 0.25, 2, tm);
    }
}

// Does the t = tmax drift converge once the filament is resolved?  At 128^2
// the structure at t = 2 is ~2 cells thick, i.e. exactly at the limit the
// frozen-band redistance can represent.  Refining at a cfl where the time
// error is not dominant answers it.
void runDeep(int rank) {
    if (rank == 0) std::printf("--- t = 2 convergence (cfl 0.25, ls_iterations 2) ---\n");
    for (int nx : {128, 256, 384, 512}) sweepCase(rank, "deep", nx, 0.25, 2, 2.0);

    // Second question: the Spec's default cfl is 0.5 and the integrator is
    // forward Euler.  Holding the mesh fixed at 256^2 (where cfl 0.25 already
    // gives 8.8%) and walking cfl down separates the temporal error from the
    // spatial one.  A smooth O(dt) fall means "accuracy"; a cliff between 0.5
    // and 0.35 means the FE + WENO5 pair sits outside its stability limit at
    // the default cfl.
    if (rank == 0) {
        std::printf("--- forward-Euler cfl sweep at 256^2 (ls_iterations 2, t = 2) ---\n");
    }
    for (double c : {0.50, 0.40, 0.35, 0.30, 0.25}) sweepCase(rank, "deep-cfl", 256, c, 2, 2.0);
}

// Third question, and the one the human needs answered before the Spec's 5%
// bar can be judged: is there ANY affordable (mesh, cfl) pair at which the
// mandated scheme meets 5% at t = tmax = 2?  Push both knobs at once.
void runBest(int rank) {
    if (rank == 0) std::printf("--- best affordable configurations at t = 2 (ls_iterations 2) ---\n");
    sweepCase(rank, "best", 256, 0.125, 2, 2.0);
    sweepCase(rank, "best", 384, 0.125, 2, 2.0);
    sweepCase(rank, "best", 512, 0.125, 2, 2.0);
}

// U9's reversal deliverable: with reverse_time = true the velocity is scaled by
// cos(pi t / tmax), so the t = tmax field must return to the initial circle.
// One number is not evidence that what remains is *discretization* error --
// it has to fall with h.  This prints the area drift and the max-norm
// departure from phi(t = 0) for a refinement sequence.
void reverseCase(int rank, int nx, double cfl) {
    Params p;
    p.nx = nx;
    p.ny = nx;
    p.tmax = 2.0;
    p.cfl = cfl;
    p.ls_iterations = 2;
    p.reverse_time = true;
    p.verbosity = 0;

    Grid g;
    std::string err;
    if (!g.create(p, MPI_COMM_WORLD, err)) return;
    const RunResult r = runToTmax(g, p);
    if (rank == 0) {
        std::printf("reverse nx %4d cfl %6.3f | steps %5d | area drift %7.3f%% (sharp %7.3f%%)"
                    "  max|phi(tmax)-phi(0)| %.4e (= %7.3f h)\n",
                    nx, cfl, r.steps, 100.0 * r.drift_smoothed, 100.0 * r.drift_sharp,
                    r.max_norm_vs_initial, r.max_norm_vs_initial / g.dx());
        std::fflush(stdout);
    }
    g.destroy();
}

void runReverse(int rank) {
    if (rank == 0) std::printf("--- reversal mode (reverse_time = true), t = tmax = 2 ---\n");
    for (int nx : {64, 128, 256}) reverseCase(rank, nx, 0.5);
    for (int nx : {64, 128, 256}) reverseCase(rank, nx, 0.125);
}

void report(int rank, const char* label, const RunResult& r) {
    if (rank != 0) return;
    std::printf("test_conservation: %s\n", label);
    std::printf("  steps %d to t = %.6f\n", r.steps, r.time);
    std::printf("  area (Spec, smoothed H over 1.5 cells): %.9g -> %.9g   drift %.4f%%\n",
                r.a0.smoothed, r.a1.smoothed, 100.0 * r.drift_smoothed);
    std::printf("  area (sharp sign count)               : %.9g -> %.9g   drift %.4f%%\n",
                r.a0.sharp, r.a1.sharp, 100.0 * r.drift_sharp);
    std::printf("  phi in [%.6f, %.6f]   max|div u| %.3e   sign flips %ld\n", r.phi_min,
                r.phi_max, r.max_divergence, r.sign_flips);
    std::printf("  |phi(t_end) - phi(t=0)| max-norm      : %.6e\n", r.max_norm_vs_initial);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    bool sweep = false;
    bool probe = false;
    bool deep = false;
    bool best = false;
    bool reverse = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--sweep") sweep = true;
        if (std::string(argv[i]) == "--probe") probe = true;
        if (std::string(argv[i]) == "--deep") deep = true;
        if (std::string(argv[i]) == "--best") best = true;
        if (std::string(argv[i]) == "--reverse") reverse = true;
    }
    if (sweep || probe || deep || best || reverse) {
        if (sweep) runSweep(rank);
        if (probe) runProbe(rank);
        if (deep) runDeep(rank);
        if (best) runBest(rank);
        if (reverse) runReverse(rank);
        MPI_Finalize();
        return 0;
    }

    int failures = 0;
    {
        // Defaults identical to params/bubble.ini, including the two
        // human-authorised deviations from the Spec's stated values (U9 round,
        // recorded in dev/tasks/init-project/log.md and in bubble.ini itself):
        //   cfl   0.5 -> 0.05  forward Euler + WENO5 is unstable at cfl = 0.5
        //   nx,ny 128 -> 256   the t = 2 filament is sub-grid at 128^2
        // The Spec's 5% bar itself is unchanged.
        Params p;
        p.nx = 256;
        p.ny = 256;
        p.tmax = 2.0;
        p.cfl = 0.05;
        p.ls_iterations = 2;
        p.verbosity = 0;

        Grid g;
        std::string err;
        if (!g.create(p, MPI_COMM_WORLD, err)) {
            if (rank == 0) std::printf("test_conservation: FAIL grid: %s\n", err.c_str());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // --- the Spec's row: baseline run to tmax ---------------------------
        const RunResult base = runToTmax(g, p);
        report(rank, "baseline (reverse_time = false), 256^2, cfl 0.05", base);

        // Invariant 6 must hold at every step of the run as well.
        if (base.max_divergence > 1.0e-12) {
            if (rank == 0) {
                std::printf("test_conservation: FAIL invariant 6: max|div u| = %.3e > 1e-12\n",
                            base.max_divergence);
            }
            ++failures;
        }

        if (!(base.drift_smoothed <= 0.05)) {
            if (rank == 0) {
                std::printf("test_conservation: FAIL area drift %.4f%% exceeds the Spec bar of "
                            "5%% (area %.9g -> %.9g)\n",
                            100.0 * base.drift_smoothed, base.a0.smoothed, base.a1.smoothed);
            }
            ++failures;
        }

        // --- diagnostic, not a gate: the Rider-Kothe reversal run -----------
        Params q = p;
        q.reverse_time = true;
        const RunResult rev = runToTmax(g, q);
        report(rank, "reversal mode (reverse_time = true), 256^2, cfl 0.05", rev);

        g.destroy();
    }

    if (rank == 0) {
        std::printf("test_conservation: %s\n", failures == 0 ? "PASS" : "FAIL");
        std::fflush(stdout);
    }
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
