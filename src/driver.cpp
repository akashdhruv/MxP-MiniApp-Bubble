// driver.cpp -- time-evolution loop (U6).  See driver.h for the reference.

#include "driver.h"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "advect.h"
#include "init.h"
#include "io.h"
#include "redistance.h"

double computeTimestep(const Params& p, const Grid& g, double umax, double vmax, double time) {
    double dt = 0.0;
    if (p.fixed_dt > 0.0) {
        dt = p.fixed_dt;
    } else {
        const double denom = umax / g.dx() + vmax / g.dy();
        // A zero velocity field (possible only in pathological parameter
        // choices) must not produce an infinite step.
        dt = (denom > 0.0) ? (p.cfl / denom) : (p.tmax - time);
    }
    const double remaining = p.tmax - time;
    if (dt > remaining) dt = remaining;
    return dt;
}

long advanceOneStep(const Grid& g, const Params& p, std::vector<double>& phi,
                    const std::vector<double>& u, const std::vector<double>& v, double dt,
                    std::vector<double>& rhs, std::vector<double>& phiOrig) {
    if (rhs.size() != phi.size()) rhs.assign(phi.size(), 0.0);

    // WENO5 accumulates into rhs (the Fortran does), so zero it first.
    std::fill(rhs.begin(), rhs.end(), 0.0);
    advectWeno(g, phi, u, v, rhs);

    // Forward Euler (Stencils_integrateEuler, no source term).
    for (int j = 0; j < g.ny(); ++j) {
        for (int i = 0; i < g.nx(); ++i) {
            const int k = g.index(i, j);
            phi[k] += dt * rhs[k];
        }
    }
    g.exchange(phi);

    // mph_lsIt redistance iterations; guard cells refreshed between them.
    return redistance(g, phi, phiOrig, p.ls_iterations);
}

bool runSimulation(const Grid& g, const Params& p, std::vector<double>& dfun, RunSummary& summary,
                   std::string& err) {
    std::vector<double> u = g.allocate();
    std::vector<double> v = g.allocate();
    std::vector<double> rhs = g.allocate();
    std::vector<double> phiOrig = g.allocate();

    // Unscaled maximum face speeds: the velocity field is steady, and the
    // optional reverse_time factor cos(pi t / tmax) only shrinks it, so these
    // give a dt that is stable for the whole run and independent of t.
    {
        Params q = p;
        q.reverse_time = false;
        computeFaceVelocity(g, q, 0.0, u, v);
    }
    double umax = 0.0;
    double vmax = 0.0;
    maxFaceSpeeds(g, u, v, umax, vmax);

    if (!ensureOutputDir(g, p, err)) return false;
    if (!writeHistoryHeader(g, p, err)) return false;

    double time = 0.0;
    int step = 0;

    // Initial state: velocity at t = 0 (scaled, if reverse_time), output.
    computeFaceVelocity(g, p, time, u, v);
    Diagnostics diag = computeDiagnostics(g, dfun, u, v, 0L);
    summary.area_initial = diag.area;
    summary.max_divergence = diag.max_divergence;
    if (!writeFields(g, p, step, time, dfun, u, v, err)) return false;
    ++summary.plots_written;
    if (!appendHistory(g, p, step, time, 0.0, diag, err)) return false;
    ++summary.history_rows;

    const double tEps = 1.0e-12 * p.tmax;
    while (time < p.tmax - tEps) {
        if (p.max_steps > 0 && step >= p.max_steps) break;

        const double dt = computeTimestep(p, g, umax, vmax, time);
        if (dt <= 0.0) break;

        // Velocity is recomputed every step, mirroring the reference driver.
        computeFaceVelocity(g, p, time, u, v);

        const long flips = advanceOneStep(g, p, dfun, u, v, dt, rhs, phiOrig);
        summary.sign_flips_total += flips;

        time += dt;
        ++step;
        summary.dt_last = dt;

        const bool lastStep =
            (time >= p.tmax - tEps) || (p.max_steps > 0 && step >= p.max_steps);
        const bool wantHist = (p.hist_interval > 0 && step % p.hist_interval == 0) || lastStep;
        const bool wantPlot = (p.plot_interval > 0 && step % p.plot_interval == 0) || lastStep;

        if (wantHist || wantPlot) {
            diag = computeDiagnostics(g, dfun, u, v, flips);
            if (diag.max_divergence > summary.max_divergence) {
                summary.max_divergence = diag.max_divergence;
            }
        }
        if (wantHist) {
            if (!appendHistory(g, p, step, time, dt, diag, err)) return false;
            ++summary.history_rows;
        }
        if (wantPlot) {
            if (!writeFields(g, p, step, time, dfun, u, v, err)) return false;
            ++summary.plots_written;
        }
        if (p.verbosity > 1 && g.rank() == 0 && (step % 100 == 0 || lastStep)) {
            std::printf("bubble: step %6d  t %.6f  dt %.3e  area %.9f  flips %ld\n", step, time,
                        dt, diag.area, flips);
            std::fflush(stdout);
        }
    }

    summary.steps = step;
    summary.time = time;
    summary.area_final = diag.area;
    summary.area_drift = (summary.area_initial > 0.0)
                             ? std::fabs(diag.area - summary.area_initial) / summary.area_initial
                             : 0.0;
    return true;
}
