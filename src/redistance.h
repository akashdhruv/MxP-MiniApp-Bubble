// redistance.h -- level-set redistance (U4).
//
// Ported from
//   Flash-X-Development/Flash-X/source/numericalTools/Stencils/
//     StencilsLevelset/Stencils_lsRedistance2d.F90
// with the driver logic of
//   Flash-X-Development/Flash-X/source/physics/Multiphase/MultiphaseMain/
//     Multiphase_redistance.F90
// (snapshot DFUN into HDN0 on iteration 1, lsDT = |cell diagonal| / 5).
// Spec: dev/tasks/init-project/spec.md, "Redistance".
//
// Scheme, verbatim from the Fortran:
//   sgn   = phi_orig / |phi_orig + eps|,  eps = 1e-15   (sign from the
//           *original* field, never from the relaxing one)
//   phio  = phi                                        (Jacobi, not
//           Gauss-Seidel: every cell reads the previous iterate)
//   interface cell  (phio(i,j) * phio(neighbour) < 0 for any of the four
//           neighbours)  ->  phi = phi_orig, frozen, not relaxed
//   otherwise  a+- = max/min(sm - sxl, 0)/dx,  b+- = max/min(sxr - sm, 0)/dx
//              c+- , d+-  likewise in y
//              phio > 0 : agf = sqrt(max(ap^2,bn^2) + max(cp^2,dn^2)) - 1
//              phio < 0 : agf = sqrt(max(an^2,bp^2) + max(cn^2,dp^2)) - 1
//              phio = 0 : agf = 0
//              phi = phio - lsDT * sgn * agf
//   a sign flip of phi relative to phi_orig is a warning: the Fortran prints
//   one line per cell, the Spec asks for a count reported in history.csv.
#ifndef BUBBLE_REDISTANCE_H
#define BUBBLE_REDISTANCE_H

#include <vector>

#include "grid.h"

// Pseudo-timestep of the redistance iteration: sqrt(dx^2 + dy^2) / 5
// (Multiphase_redistance.F90: minCellDiag/5, with del(DIR_Z) = 0 in 2D).
double lsPseudoTimestep(const Grid& g);

// |grad phi| at owned cell (i,j) by the first-order Godunov upwind choice
// above (the Fortran's `agf + 1`).  Reads phi(i+-1, j), phi(i, j+-1), so the
// caller must have current guard cells.  Exposed for diagnostics and tests.
double godunovGradMag(const Grid& g, const std::vector<double>& phi, int i, int j);

// One redistance iteration over the owned cells.  `phi` is updated in place
// from a snapshot of itself (Jacobi); `phiOrig` supplies both the sign and the
// frozen interface values.  Guard cells of `phi` must be current on entry.
// Returns the number of *locally owned* cells whose sign differs from
// phi_orig after the update.
long redistanceIteration(const Grid& g, std::vector<double>& phi,
                         const std::vector<double>& phiOrig, double lsDT);

// `iterations` redistance iterations (Spec: mph_lsIt = 2 per timestep).
// On iteration 1 the snapshot phi_orig = phi is taken, exactly as
// Multiphase_redistance does; `phiOrig` is resized if needed and returned to
// the caller so the diagnostics can use it.  Guard cells are refreshed before
// every iteration -- including between iterations, which is the Spec's named
// silent trap -- and once more on exit, so `phi` leaves with valid halos.
// Returns the global (MPI_Allreduce'd) sign-flip count of the last iteration.
long redistance(const Grid& g, std::vector<double>& phi, std::vector<double>& phiOrig,
                int iterations);

#endif  // BUBBLE_REDISTANCE_H
