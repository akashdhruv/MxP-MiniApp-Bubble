// driver.h -- time-evolution loop (U6).
//
// Ported from
//   Flash-X-Development/Flash-X/source/Simulation/SimulationMain/incompFlow/
//     DeformingBubble/Driver_evolveAll.F90-mc
// (per step: recompute the prescribed velocity, advect the level set, integrate
// in time, redistance, then diagnostics and output) and specified in
// dev/tasks/init-project/spec.md, "Advection", "Time integration",
// "Redistance", "Timestep".
//
// One step, in the reference's order:
//   1. face velocity u,v at the current time (steady, but recomputed as the
//      reference driver does; with reverse_time it is scaled by cos(pi t/tmax))
//   2. rhs = 0; WENO5 flux-difference advection accumulates into rhs
//   3. forward Euler:  phi <- phi + dt * rhs     (Stencils_integrateEuler)
//   4. ls_iterations = mph_lsIt = 2 redistance iterations, guard cells
//      refreshed between them
#ifndef BUBBLE_DRIVER_H
#define BUBBLE_DRIVER_H

#include <string>
#include <vector>

#include "grid.h"
#include "io.h"
#include "params.h"

// CFL timestep: dt = cfl / (umax/dx + vmax/dy), clipped so the step lands
// exactly on tmax; `fixed_dt` overrides the estimate (bit-comparison runs).
// `umax`/`vmax` are the *unscaled* maximum face speeds, so reverse_time (which
// only multiplies the field by cos(pi t/tmax), |.| <= 1) cannot make the step
// unstable or make dt blow up as the field passes through zero.
double computeTimestep(const Params& p, const Grid& g, double umax, double vmax, double time);

// One timestep on an already-halo-filled `phi`: WENO5 advection + forward
// Euler + ls_iterations redistance iterations.  `u`,`v` are the face
// velocities for this step; `rhs`, `phiOrig` are scratch supplied by the
// caller so the loop does not reallocate.  Returns the global sign-flip count
// reported by the redistance.
long advanceOneStep(const Grid& g, const Params& p, std::vector<double>& phi,
                    const std::vector<double>& u, const std::vector<double>& v, double dt,
                    std::vector<double>& rhs, std::vector<double>& phiOrig);

// Summary of a completed run.
struct RunSummary {
    int steps = 0;
    double time = 0.0;
    double dt_last = 0.0;
    double area_initial = 0.0;
    double area_final = 0.0;
    double area_drift = 0.0;      // |final - initial| / initial
    double max_divergence = 0.0;  // over the whole run
    long sign_flips_total = 0;
    int plots_written = 0;
    int history_rows = 0;
};

// Run the loop from t = 0 to p.tmax (or p.max_steps steps, whichever comes
// first), writing out/history.csv every p.hist_interval steps and the
// .vti/.pvti set every p.plot_interval steps, plus the initial and final
// states.  `dfun` must hold the initial condition with valid guard cells and
// is left holding the final field.  Returns false with `err` filled if output
// cannot be written.
bool runSimulation(const Grid& g, const Params& p, std::vector<double>& dfun, RunSummary& summary,
                   std::string& err);

#endif  // BUBBLE_DRIVER_H
