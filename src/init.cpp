// init.cpp -- see init.h.
//
// Reference read for this unit:
//   Flash-X-Development/Flash-X/source/Simulation/SimulationMain/incompFlow/
//     DeformingBubble/Simulation_initBlock.F90
//
// The two face loops there read, verbatim:
//
//   facexData(VELC_FACE_VAR,i,j,k) = -((sin(pi*xi))**2)*(cos(2*pi*(yi+del(JAXIS)/2)) -
//                                     cos(2*pi*(yi-del(JAXIS)/2)))/(2*pi*del(JAXIS))
//   faceyData(VELC_FACE_VAR,i,j,k) =  ((sin(pi*yi))**2)*(cos(2*pi*(xi+del(IAXIS)/2)) -
//                                     cos(2*pi*(xi-del(IAXIS)/2)))/(2*pi*del(IAXIS))
//
// with xi an x-FACE coordinate and yi a y-CENTER coordinate for facex, and the
// other way round for facey; and the level set
//
//   solnData(DFUN_VAR,i,j,k) = 0.1 - sqrt((xi-xb)**2 + (yi-yb)**2)
//
// at cell centres.  That is what is implemented below.

#include "init.h"

#include <cmath>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

}  // namespace

double levelSetAnalytic(const Params& p, double x, double y) {
    const double rx = x - p.bubble_x;
    const double ry = y - p.bubble_y;
    return p.bubble_r - std::sqrt(rx * rx + ry * ry);
}

void initLevelSet(const Grid& g, const Params& p, std::vector<double>& dfun) {
    if (dfun.size() != g.size()) dfun.assign(g.size(), 0.0);

    for (int j = 0; j < g.ny(); ++j) {
        const double y = g.yCenter(j);
        for (int i = 0; i < g.nx(); ++i) {
            dfun[g.index(i, j)] = levelSetAnalytic(p, g.xCenter(i), y);
        }
    }
    // Guard cells: neighbour values across a rank boundary, Neumann at a wall.
    g.exchange(dfun);
}

void computeFaceVelocity(const Grid& g, const Params& p, double time, std::vector<double>& u,
                         std::vector<double>& v) {
    if (u.size() != g.size()) u.assign(g.size(), 0.0);
    if (v.size() != g.size()) v.assign(g.size(), 0.0);

    const double dx = g.dx();
    const double dy = g.dy();
    // Spec "Optional validation mode": steady field unless reverse_time.
    const double scale = p.reverse_time ? std::cos(kPi * time / p.tmax) : 1.0;

    const int ng = g.ng();

    // u: x-face coordinate in x, cell-centre coordinate in y.
    for (int j = -ng; j < g.ny() + ng; ++j) {
        const double yc = g.yCenter(j);
        const double dcos =
            std::cos(2.0 * kPi * (yc + 0.5 * dy)) - std::cos(2.0 * kPi * (yc - 0.5 * dy));
        for (int i = -ng; i < g.nx() + ng; ++i) {
            const double xf = g.xFace(i);
            const double s = std::sin(kPi * xf);
            u[g.index(i, j)] = scale * (-(s * s) * dcos / (2.0 * kPi * dy));
        }
    }

    // v: cell-centre coordinate in x, y-face coordinate in y.
    for (int j = -ng; j < g.ny() + ng; ++j) {
        const double yf = g.yFace(j);
        const double s = std::sin(kPi * yf);
        const double s2 = s * s;
        for (int i = -ng; i < g.nx() + ng; ++i) {
            const double xc = g.xCenter(i);
            const double dcos =
                std::cos(2.0 * kPi * (xc + 0.5 * dx)) - std::cos(2.0 * kPi * (xc - 0.5 * dx));
            v[g.index(i, j)] = scale * (s2 * dcos / (2.0 * kPi * dx));
        }
    }
}

double maxDivergence(const Grid& g, const std::vector<double>& u, const std::vector<double>& v) {
    double local = 0.0;
    for (int j = 0; j < g.ny(); ++j) {
        for (int i = 0; i < g.nx(); ++i) {
            const double div = (u[g.index(i + 1, j)] - u[g.index(i, j)]) / g.dx() +
                               (v[g.index(i, j + 1)] - v[g.index(i, j)]) / g.dy();
            const double a = std::fabs(div);
            if (a > local) local = a;
        }
    }
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, g.comm());
    return global;
}

void maxFaceSpeeds(const Grid& g, const std::vector<double>& u, const std::vector<double>& v,
                   double& umax, double& vmax) {
    double lu = 0.0;
    double lv = 0.0;
    for (int j = 0; j < g.ny(); ++j) {
        for (int i = 0; i <= g.nx(); ++i) {
            const double a = std::fabs(u[g.index(i, j)]);
            if (a > lu) lu = a;
        }
    }
    for (int j = 0; j <= g.ny(); ++j) {
        for (int i = 0; i < g.nx(); ++i) {
            const double a = std::fabs(v[g.index(i, j)]);
            if (a > lv) lv = a;
        }
    }
    double in[2] = {lu, lv};
    double out[2] = {0.0, 0.0};
    MPI_Allreduce(in, out, 2, MPI_DOUBLE, MPI_MAX, g.comm());
    umax = out[0];
    vmax = out[1];
}
