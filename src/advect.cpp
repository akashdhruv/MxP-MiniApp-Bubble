// advect.cpp -- WENO5 advection kernel, ported from
//   Flash-X-Development/Flash-X/source/numericalTools/Stencils/
//     StencilsAdvection/Stencils_advectWeno.F90
//
// The Fortran is written out twice per direction (a "r" high-face block and an
// "l" low-face block) and twice again for the sign of the face velocity, four
// copies of the same algebra per direction.  The only differences are the
// stencil shift and the linear weights, so the algebra lives once in
// wenoFaceValue() here.  The correspondence, face m being the *low* face of
// cell m (MAC convention of the Spec, face(center) = 1 in the Fortran call):
//
//   velocity > 0 ("downwind" branch in the Fortran comments):
//       s1..s5 = phi(m-3), phi(m-2), phi(m-1), phi(m), phi(m+1)
//   velocity <= 0 ("upwind" branch):
//       s1..s5 = phi(m-2), phi(m-1), phi(m), phi(m+1), phi(m+2)
//
// Check against the Fortran: the high face of cell i is m = i+1, so the
// positive branch reads phi(i-2..i+2) and the negative branch phi(i-1..i+3),
// which is exactly lines 75-79 and 102-106 of the Fortran; the low face is
// m = i, giving phi(i-3..i+1) and phi(i-2..i+2), lines 131-135 and 158-162.
//
// Coefficients are transcribed verbatim: 13/12 and 1/4 in the smoothness
// indicators, eps = 1e-15, exponent 2 on (eps + rIS), linear weights
// (1/10, 6/10, 3/10) for a positive face velocity and (3/10, 6/10, 1/10) for a
// negative one, and the three candidate reconstructions
//   positive:  (2 s1 - 7 s2 + 11 s3)/6, (-s2 + 5 s3 + 2 s4)/6, (2 s3 + 5 s4 - s5)/6
//   negative:  (-s1 + 5 s2 + 2 s3)/6,   (2 s2 + 5 s3 - s4)/6,  (11 s3 - 7 s4 + 2 s5)/6

#include "advect.h"

#include <cstddef>

namespace {

constexpr double kEps = 1.0e-15;  // Fortran: eps = 1e-15

inline double sqr(double x) { return x * x; }

}  // namespace

double wenoFaceValue(double s1, double s2, double s3, double s4, double s5, bool positive) {
    // Smoothness indicators (identical in both branches).
    const double rIS1 = (13.0 / 12.0) * sqr(s1 - 2.0 * s2 + s3) +
                        (1.0 / 4.0) * sqr(s1 - 4.0 * s2 + 3.0 * s3);
    const double rIS2 = (13.0 / 12.0) * sqr(s2 - 2.0 * s3 + s4) +
                        (1.0 / 4.0) * sqr(s2 - s4);
    const double rIS3 = (13.0 / 12.0) * sqr(s3 - 2.0 * s4 + s5) +
                        (1.0 / 4.0) * sqr(3.0 * s3 - 4.0 * s4 + s5);

    // Linear weights flip with the sign of the face velocity.
    const double d1 = positive ? (1.0 / 10.0) : (3.0 / 10.0);
    const double d2 = 6.0 / 10.0;
    const double d3 = positive ? (3.0 / 10.0) : (1.0 / 10.0);

    const double aT1 = d1 / sqr(kEps + rIS1);
    const double aT2 = d2 / sqr(kEps + rIS2);
    const double aT3 = d3 / sqr(kEps + rIS3);

    const double aSum = aT1 + aT2 + aT3;
    const double a1 = aT1 / aSum;
    const double a2 = aT2 / aSum;
    const double a3 = aT3 / aSum;

    double fT1;
    double fT2;
    double fT3;
    if (positive) {
        fT1 = (2.0 / 6.0) * s1 - (7.0 / 6.0) * s2 + (11.0 / 6.0) * s3;
        fT2 = -(1.0 / 6.0) * s2 + (5.0 / 6.0) * s3 + (2.0 / 6.0) * s4;
        fT3 = (2.0 / 6.0) * s3 + (5.0 / 6.0) * s4 - (1.0 / 6.0) * s5;
    } else {
        fT1 = -(1.0 / 6.0) * s1 + (5.0 / 6.0) * s2 + (2.0 / 6.0) * s3;
        fT2 = (2.0 / 6.0) * s2 + (5.0 / 6.0) * s3 - (1.0 / 6.0) * s4;
        fT3 = (11.0 / 6.0) * s3 - (7.0 / 6.0) * s4 + (2.0 / 6.0) * s5;
    }

    return a1 * fT1 + a2 * fT2 + a3 * fT3;
}

void advectWeno(const Grid& g, const std::vector<double>& phi, const std::vector<double>& u,
                const std::vector<double>& v, std::vector<double>& rhs) {
    if (rhs.size() != g.size()) rhs.assign(g.size(), 0.0);

    const int nx = g.nx();
    const int ny = g.ny();
    const double dxi = 1.0 / g.dx();
    const double dyi = 1.0 / g.dy();

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            // face(center) = 1 in the Fortran: ul/ur and vl/vr are the MAC
            // face velocities themselves, no averaging.
            const double ul = u[g.index(i, j)];
            const double ur = u[g.index(i + 1, j)];
            const double vl = v[g.index(i, j)];
            const double vr = v[g.index(i, j + 1)];

            // ---- x direction -------------------------------------------------
            double flx;
            double frx;
            if (ur > 0.0) {
                frx = wenoFaceValue(phi[g.index(i - 2, j)], phi[g.index(i - 1, j)],
                                    phi[g.index(i, j)], phi[g.index(i + 1, j)],
                                    phi[g.index(i + 2, j)], true);
            } else {
                frx = wenoFaceValue(phi[g.index(i - 1, j)], phi[g.index(i, j)],
                                    phi[g.index(i + 1, j)], phi[g.index(i + 2, j)],
                                    phi[g.index(i + 3, j)], false);
            }
            if (ul > 0.0) {
                flx = wenoFaceValue(phi[g.index(i - 3, j)], phi[g.index(i - 2, j)],
                                    phi[g.index(i - 1, j)], phi[g.index(i, j)],
                                    phi[g.index(i + 1, j)], true);
            } else {
                flx = wenoFaceValue(phi[g.index(i - 2, j)], phi[g.index(i - 1, j)],
                                    phi[g.index(i, j)], phi[g.index(i + 1, j)],
                                    phi[g.index(i + 2, j)], false);
            }

            // ---- y direction -------------------------------------------------
            double fly;
            double fry;
            if (vr > 0.0) {
                fry = wenoFaceValue(phi[g.index(i, j - 2)], phi[g.index(i, j - 1)],
                                    phi[g.index(i, j)], phi[g.index(i, j + 1)],
                                    phi[g.index(i, j + 2)], true);
            } else {
                fry = wenoFaceValue(phi[g.index(i, j - 1)], phi[g.index(i, j)],
                                    phi[g.index(i, j + 1)], phi[g.index(i, j + 2)],
                                    phi[g.index(i, j + 3)], false);
            }
            if (vl > 0.0) {
                fly = wenoFaceValue(phi[g.index(i, j - 3)], phi[g.index(i, j - 2)],
                                    phi[g.index(i, j - 1)], phi[g.index(i, j)],
                                    phi[g.index(i, j + 1)], true);
            } else {
                fly = wenoFaceValue(phi[g.index(i, j - 2)], phi[g.index(i, j - 1)],
                                    phi[g.index(i, j)], phi[g.index(i, j + 1)],
                                    phi[g.index(i, j + 2)], false);
            }

            rhs[g.index(i, j)] += -(frx * ur - flx * ul) * dxi - (fry * vr - fly * vl) * dyi;
        }
    }
}
