// advect.h -- WENO5 advection kernel (U3).
//
// Ported from
//   Flash-X-Development/Flash-X/source/numericalTools/Stencils/
//     StencilsAdvection/Stencils_advectWeno.F90
// called with face(center) = 1 (cell-centred scalar, MAC face velocities),
// NDIM = 2 so the z term is zero.  Spec: dev/tasks/init-project/spec.md,
// "Advection".
//
// Flux-difference form, exactly the Fortran's last statement:
//
//   rhs(i,j) -= ( frx*u(i+1,j) - flx*u(i,j) )/dx
//             + ( fry*v(i,j+1) - fly*v(i,j) )/dy
//
// with frx/flx/fry/fly the WENO5 reconstructions of phi at the four faces of
// cell (i,j).  Note the Fortran *accumulates* into rhs; advectWeno() does the
// same, so the caller zeroes (or pre-loads) rhs.
#ifndef BUBBLE_ADVECT_H
#define BUBBLE_ADVECT_H

#include <vector>

#include "grid.h"

// WENO5 reconstruction of phi at one face from the five-point stencil
// s1..s5, with `positive` selecting the upwind branch on the sign of the face
// velocity (Fortran: `if (ur .gt. 0)`).  Linear weights are (1/10, 6/10, 3/10)
// for a positive face velocity and (3/10, 6/10, 1/10) for a negative one;
// smoothness indicators use the 13/12 and 1/4 coefficients, eps = 1e-15 and
// exponent 2.  Exposed so the tests can exercise the reconstruction directly.
double wenoFaceValue(double s1, double s2, double s3, double s4, double s5, bool positive);

// Add the WENO5 advection term to `rhs` over all owned cells.
//
// The caller must have refreshed the guard cells of `phi` (3 layers) before
// calling: the stencil reaches i-3 .. i+3.  `u` and `v` are MAC face
// velocities with u(i,j) the low x-face of cell (i,j) and v(i,j) the low
// y-face; the high faces u(nx,j), v(i,ny) are read as well.  `rhs` is resized
// (and the new cells zeroed) if it does not already match the grid.
void advectWeno(const Grid& g, const std::vector<double>& phi, const std::vector<double>& u,
                const std::vector<double>& v, std::vector<double>& rhs);

#endif  // BUBBLE_ADVECT_H
