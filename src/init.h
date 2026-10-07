// init.h -- initial condition and prescribed face-velocity field.
//
// Ported from
//   Flash-X-Development/Flash-X/source/Simulation/SimulationMain/incompFlow/
//     DeformingBubble/Simulation_initBlock.F90
// and specified in dev/tasks/init-project/spec.md, "Physics and numerics
// contract" ("Initial condition", "Velocity field").
//
// Staggering (MAC, Spec "Staggering"): dfun lives at cell centres, u on
// x-faces with u(i,j) the *low* x-face of cell (i,j), v on y-faces likewise.
// Face fields use the same Grid::index() mapping and allocation size, so the
// high face of the last owned cell, u(nx,j) / v(i,ny), is addressable.
#ifndef BUBBLE_INIT_H
#define BUBBLE_INIT_H

#include <vector>

#include "grid.h"
#include "params.h"

// phi = bubble_r - |x - x_bubble|  (positive inside), exactly the Flash-X
// `0.1 - sqrt((xi-xb)**2 + (yi-yb)**2)` with the radius taken from params.
// Owned cells are set analytically; guard cells are then filled by the halo
// exchange, i.e. neighbour data across a rank boundary and zero-gradient
// (Neumann) extrapolation at a physical wall.  `dfun` is resized if needed.
void initLevelSet(const Grid& g, const Params& p, std::vector<double>& dfun);

// Analytic value of the initial level set at a point (used by the tests and
// by the reversal-mode error metric).
double levelSetAnalytic(const Params& p, double x, double y);

// Discrete streamfunction velocity, psi = sin^2(pi x) sin^2(pi y) / pi,
// differenced so that the discrete divergence is zero to machine precision:
//
//   u(i,j) = -sin^2(pi x_i) (cos(2 pi (y_j+dy/2)) - cos(2 pi (y_j-dy/2)))/(2 pi dy)
//   v(i,j) = +sin^2(pi y_j) (cos(2 pi (x_i+dx/2)) - cos(2 pi (x_i-dx/2)))/(2 pi dx)
//
// Do NOT substitute the analytic curl: the differenced form is what makes
// invariant 6 hold.  Evaluated analytically everywhere, guard regions
// included (Spec, "Boundary conditions").  With reverse_time the whole field
// is scaled by cos(pi t / tmax), which cannot break the divergence property.
// `u` and `v` are resized if needed.
void computeFaceVelocity(const Grid& g, const Params& p, double time, std::vector<double>& u,
                         std::vector<double>& v);

// max |(u(i+1,j)-u(i,j))/dx + (v(i,j+1)-v(i,j))/dy| over all owned cells,
// MPI_Allreduce'd over the Cartesian communicator (invariant 6).
double maxDivergence(const Grid& g, const std::vector<double>& u, const std::vector<double>& v);

// max |u| and max |v| over owned faces, reduced over the communicator;
// used by the CFL timestep in the driver.
void maxFaceSpeeds(const Grid& g, const std::vector<double>& u, const std::vector<double>& v,
                   double& umax, double& vmax);

#endif  // BUBBLE_INIT_H
