// redistance.cpp -- see redistance.h for the provenance of every line of the
// scheme (Stencils_lsRedistance2d.F90 + Multiphase_redistance.F90).

#include "redistance.h"

#include <cmath>

namespace {

constexpr double kEps = 1.0e-15;  // Fortran: eps = 1E-15

// Fortran SIGN(1.0, x): +1 for x >= 0 (including -0.0 is treated as the IEEE
// sign in Fortran, but the Fortran code compares two SIGN results and the
// level set never holds -0.0 here), -1 for x < 0.
inline double signOf(double x) { return (x < 0.0) ? -1.0 : 1.0; }

inline double maxd(double a, double b) { return (a > b) ? a : b; }
inline double mind(double a, double b) { return (a < b) ? a : b; }

}  // namespace

double lsPseudoTimestep(const Grid& g) {
    const double minCellDiag = std::sqrt(g.dx() * g.dx() + g.dy() * g.dy());
    return minCellDiag / 5.0;
}

double godunovGradMag(const Grid& g, const std::vector<double>& phi, int i, int j) {
    const double dx = g.dx();
    const double dy = g.dy();
    const double sm = phi[g.index(i, j)];
    const double sxl = phi[g.index(i - 1, j)];
    const double sxr = phi[g.index(i + 1, j)];
    const double syl = phi[g.index(i, j - 1)];
    const double syr = phi[g.index(i, j + 1)];

    const double ap = maxd(sm - sxl, 0.0) / dx;
    const double an = mind(sm - sxl, 0.0) / dx;
    const double bp = maxd(sxr - sm, 0.0) / dx;
    const double bn = mind(sxr - sm, 0.0) / dx;
    const double cp = maxd(sm - syl, 0.0) / dy;
    const double cn = mind(sm - syl, 0.0) / dy;
    const double dp = maxd(syr - sm, 0.0) / dy;
    const double dn = mind(syr - sm, 0.0) / dy;

    if (sm > 0.0) return std::sqrt(maxd(ap * ap, bn * bn) + maxd(cp * cp, dn * dn));
    if (sm < 0.0) return std::sqrt(maxd(an * an, bp * bp) + maxd(cn * cn, dp * dp));
    return 1.0;  // agf = 0 in the Fortran, i.e. |grad phi| - 1 = 0
}

long redistanceIteration(const Grid& g, std::vector<double>& phi,
                         const std::vector<double>& phiOrig, double lsDT) {
    // Fortran: phio = phi  (whole block, guards included) -- the update is
    // Jacobi, so neighbours must come from the previous iterate.
    const std::vector<double> phio = phi;

    long flips = 0;
    for (int j = 0; j < g.ny(); ++j) {
        for (int i = 0; i < g.nx(); ++i) {
            const int c = g.index(i, j);
            const double sm = phio[c];
            const double sxl = phio[g.index(i - 1, j)];
            const double sxr = phio[g.index(i + 1, j)];
            const double syl = phio[g.index(i, j - 1)];
            const double syr = phio[g.index(i, j + 1)];

            const double orig = phiOrig[c];
            const double sgn = orig / (std::fabs(orig) + kEps);

            double value;
            if (sm * sxl < 0.0 || sm * sxr < 0.0 || sm * syl < 0.0 || sm * syr < 0.0) {
                // Interface cell: frozen to the original field.  Dropping this
                // branch lets the zero contour creep every step.
                value = orig;
            } else {
                const double ap = maxd(sm - sxl, 0.0) / g.dx();
                const double an = mind(sm - sxl, 0.0) / g.dx();
                const double bp = maxd(sxr - sm, 0.0) / g.dx();
                const double bn = mind(sxr - sm, 0.0) / g.dx();
                const double cp = maxd(sm - syl, 0.0) / g.dy();
                const double cn = mind(sm - syl, 0.0) / g.dy();
                const double dp = maxd(syr - sm, 0.0) / g.dy();
                const double dn = mind(syr - sm, 0.0) / g.dy();

                double agf;
                if (sm > 0.0) {
                    agf = std::sqrt(maxd(ap * ap, bn * bn) + maxd(cp * cp, dn * dn)) - 1.0;
                } else if (sm < 0.0) {
                    agf = std::sqrt(maxd(an * an, bp * bp) + maxd(cn * cn, dp * dp)) - 1.0;
                } else {
                    agf = 0.0;
                }
                value = sm - lsDT * (sgn * agf);
            }

            phi[c] = value;
            if (signOf(value) != signOf(orig)) ++flips;
        }
    }
    return flips;
}

long redistance(const Grid& g, std::vector<double>& phi, std::vector<double>& phiOrig,
                int iterations) {
    if (phi.size() != g.size()) phi.resize(g.size(), 0.0);
    const double lsDT = lsPseudoTimestep(g);

    long flips = 0;
    for (int it = 1; it <= iterations; ++it) {
        if (it == 1) phiOrig = phi;  // Multiphase_redistance: HDN0 = DFUN
        // Guard cells are refreshed before *every* iteration, which is the
        // "between iterations" requirement of the Spec.
        g.exchange(phi);
        flips = redistanceIteration(g, phi, phiOrig, lsDT);
    }
    if (iterations > 0) g.exchange(phi);

    long global = 0;
    MPI_Allreduce(&flips, &global, 1, MPI_LONG, MPI_SUM, g.comm());
    return global;
}
