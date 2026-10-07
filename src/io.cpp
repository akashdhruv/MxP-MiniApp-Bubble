// io.cpp -- parallel VTK ImageData writer + history CSV (U5).
//
// Spec: dev/tasks/init-project/spec.md, "Output and post-processing".
// See io.h for the file layout and the ASCII tradeoff.

#include "io.h"

#include <mpi.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <ios>
#include <sstream>

#include "init.h"

namespace {

const double kPi = 3.14159265358979323846;

std::string joinPath(const std::string& dir, const std::string& name) {
    if (dir.empty()) return name;
    if (dir[dir.size() - 1] == '/') return dir + name;
    return dir + "/" + name;
}

// 17 significant digits: enough to round-trip a double exactly, so two runs
// that agree in the field agree byte for byte in the file.
void writeDouble(std::ostream& os, double v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    os << buf;
}

}  // namespace

double smoothedHeaviside(double phi, double eps) {
    if (eps <= 0.0) return (phi > 0.0) ? 1.0 : 0.0;
    if (phi < -eps) return 0.0;
    if (phi > eps) return 1.0;
    return 0.5 * (1.0 + phi / eps + std::sin(kPi * phi / eps) / kPi);
}

double heavisideWidth(const Grid& g) {
    const double h = (g.dx() > g.dy()) ? g.dx() : g.dy();
    return 1.5 * h;
}

Diagnostics computeDiagnostics(const Grid& g, const std::vector<double>& dfun,
                               const std::vector<double>& u, const std::vector<double>& v,
                               long signFlips) {
    const double eps = heavisideWidth(g);
    const double cellArea = g.dx() * g.dy();

    double larea = 0.0;
    double lmass = 0.0;
    double lmin = 1.0e300;
    double lmax = -1.0e300;
    for (int j = 0; j < g.ny(); ++j) {
        for (int i = 0; i < g.nx(); ++i) {
            const double phi = dfun[g.index(i, j)];
            const double H = smoothedHeaviside(phi, eps);
            larea += H;
            lmass += H * phi;
            if (phi < lmin) lmin = phi;
            if (phi > lmax) lmax = phi;
        }
    }
    larea *= cellArea;
    lmass *= cellArea;

    Diagnostics d;
    MPI_Allreduce(&larea, &d.area, 1, MPI_DOUBLE, MPI_SUM, g.comm());
    MPI_Allreduce(&lmass, &d.mass, 1, MPI_DOUBLE, MPI_SUM, g.comm());
    MPI_Allreduce(&lmin, &d.phi_min, 1, MPI_DOUBLE, MPI_MIN, g.comm());
    MPI_Allreduce(&lmax, &d.phi_max, 1, MPI_DOUBLE, MPI_MAX, g.comm());
    d.max_divergence = maxDivergence(g, u, v);
    d.sign_flips = signFlips;
    return d;
}

std::string vtiName(int step, int rank) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "dfun_%06d_%04d.vti", step, rank);
    return std::string(buf);
}

std::string pvtiName(int step) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "dfun_%06d.pvti", step);
    return std::string(buf);
}

bool ensureOutputDir(const Grid& g, const Params& p, std::string& err) {
    int ok = 1;
    if (g.rank() == 0) {
        struct stat st;
        if (stat(p.outdir.c_str(), &st) == 0) {
            if (!S_ISDIR(st.st_mode)) ok = 0;
        } else if (mkdir(p.outdir.c_str(), 0755) != 0 && errno != EEXIST) {
            ok = 0;
        }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, g.comm());
    if (!ok) {
        err = "cannot create output directory '" + p.outdir + "'";
        return false;
    }
    MPI_Barrier(g.comm());
    return true;
}

bool writeHistoryHeader(const Grid& g, const Params& p, std::string& err) {
    int ok = 1;
    if (g.rank() == 0) {
        const std::string path = joinPath(p.outdir, "history.csv");
        std::ofstream f(path.c_str(), std::ios::out | std::ios::trunc);
        if (!f) {
            ok = 0;
        } else {
            f << "step,time,dt,area,mass,phi_min,phi_max,max_divergence,sign_flips\n";
            f.flush();
            if (!f) ok = 0;
        }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, g.comm());
    if (!ok) {
        err = "cannot write '" + joinPath(p.outdir, "history.csv") + "'";
        return false;
    }
    return true;
}

bool appendHistory(const Grid& g, const Params& p, int step, double time, double dt,
                   const Diagnostics& d, std::string& err) {
    int ok = 1;
    if (g.rank() == 0) {
        const std::string path = joinPath(p.outdir, "history.csv");
        std::ofstream f(path.c_str(), std::ios::out | std::ios::app);
        if (!f) {
            ok = 0;
        } else {
            std::ostringstream os;
            os << step << ',';
            writeDouble(os, time);
            os << ',';
            writeDouble(os, dt);
            os << ',';
            writeDouble(os, d.area);
            os << ',';
            writeDouble(os, d.mass);
            os << ',';
            writeDouble(os, d.phi_min);
            os << ',';
            writeDouble(os, d.phi_max);
            os << ',';
            writeDouble(os, d.max_divergence);
            os << ',' << d.sign_flips << '\n';
            f << os.str();
            f.flush();
            if (!f) ok = 0;
        }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, g.comm());
    if (!ok) {
        err = "cannot append to '" + joinPath(p.outdir, "history.csv") + "'";
        return false;
    }
    return true;
}

bool writeFields(const Grid& g, const Params& p, int step, double time,
                 const std::vector<double>& dfun, const std::vector<double>& u,
                 const std::vector<double>& v, std::string& err) {
    // Piece extent in *global point* index space: cells i0 .. i0+nx-1 span
    // points i0 .. i0+nx.
    const int ext[4] = {g.i0(), g.i0() + g.nx(), g.j0(), g.j0() + g.ny()};

    int ok = 1;
    {
        const std::string path = joinPath(p.outdir, vtiName(step, g.rank()));
        std::ofstream f(path.c_str(), std::ios::out | std::ios::trunc);
        if (!f) {
            ok = 0;
        } else {
            std::ostringstream os;
            os << "<?xml version=\"1.0\"?>\n";
            os << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
            os << "  <ImageData WholeExtent=\"" << ext[0] << ' ' << ext[1] << ' ' << ext[2] << ' '
               << ext[3] << " 0 0\" Origin=\"";
            writeDouble(os, p.xmin);
            os << ' ';
            writeDouble(os, p.ymin);
            os << " 0\" Spacing=\"";
            writeDouble(os, g.dx());
            os << ' ';
            writeDouble(os, g.dy());
            os << " 1\">\n";
            os << "    <FieldData>\n";
            os << "      <DataArray type=\"Float64\" Name=\"TIME\" NumberOfTuples=\"1\" "
                  "format=\"ascii\">";
            writeDouble(os, time);
            os << "</DataArray>\n";
            os << "    </FieldData>\n";
            os << "    <Piece Extent=\"" << ext[0] << ' ' << ext[1] << ' ' << ext[2] << ' '
               << ext[3] << " 0 0\">\n";
            os << "      <CellData Scalars=\"dfun\">\n";

            const char* names[3] = {"dfun", "u", "v"};
            for (int a = 0; a < 3; ++a) {
                os << "        <DataArray type=\"Float64\" Name=\"" << names[a]
                   << "\" format=\"ascii\">\n";
                for (int j = 0; j < g.ny(); ++j) {
                    os << "         ";
                    for (int i = 0; i < g.nx(); ++i) {
                        double val = 0.0;
                        if (a == 0) {
                            val = dfun[g.index(i, j)];
                        } else if (a == 1) {
                            // MAC -> cell centre: u(i,j) is the low x-face.
                            val = 0.5 * (u[g.index(i, j)] + u[g.index(i + 1, j)]);
                        } else {
                            val = 0.5 * (v[g.index(i, j)] + v[g.index(i, j + 1)]);
                        }
                        os << ' ';
                        writeDouble(os, val);
                    }
                    os << '\n';
                }
                os << "        </DataArray>\n";
            }
            os << "      </CellData>\n";
            os << "    </Piece>\n";
            os << "  </ImageData>\n";
            os << "</VTKFile>\n";
            f << os.str();
            f.flush();
            if (!f) ok = 0;
        }
    }

    // Rank 0 needs every piece extent for the master file.
    std::vector<int> allExt(static_cast<std::size_t>(4 * g.nprocs()), 0);
    MPI_Gather(ext, 4, MPI_INT, allExt.data(), 4, MPI_INT, 0, g.comm());

    if (g.rank() == 0 && ok) {
        const std::string path = joinPath(p.outdir, pvtiName(step));
        std::ofstream f(path.c_str(), std::ios::out | std::ios::trunc);
        if (!f) {
            ok = 0;
        } else {
            std::ostringstream os;
            os << "<?xml version=\"1.0\"?>\n";
            os << "<VTKFile type=\"PImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
            os << "  <PImageData WholeExtent=\"0 " << g.nxGlobal() << " 0 " << g.nyGlobal()
               << " 0 0\" GhostLevel=\"0\" Origin=\"";
            writeDouble(os, p.xmin);
            os << ' ';
            writeDouble(os, p.ymin);
            os << " 0\" Spacing=\"";
            writeDouble(os, g.dx());
            os << ' ';
            writeDouble(os, g.dy());
            os << " 1\">\n";
            os << "    <FieldData>\n";
            os << "      <DataArray type=\"Float64\" Name=\"TIME\" NumberOfTuples=\"1\" "
                  "format=\"ascii\">";
            writeDouble(os, time);
            os << "</DataArray>\n";
            os << "    </FieldData>\n";
            os << "    <PCellData Scalars=\"dfun\">\n";
            os << "      <PDataArray type=\"Float64\" Name=\"dfun\"/>\n";
            os << "      <PDataArray type=\"Float64\" Name=\"u\"/>\n";
            os << "      <PDataArray type=\"Float64\" Name=\"v\"/>\n";
            os << "    </PCellData>\n";
            for (int r = 0; r < g.nprocs(); ++r) {
                const int* e = &allExt[static_cast<std::size_t>(4 * r)];
                os << "    <Piece Extent=\"" << e[0] << ' ' << e[1] << ' ' << e[2] << ' ' << e[3]
                   << " 0 0\" Source=\"" << vtiName(step, r) << "\"/>\n";
            }
            os << "  </PImageData>\n";
            os << "</VTKFile>\n";
            f << os.str();
            f.flush();
            if (!f) ok = 0;
        }
    }

    int allOk = 0;
    MPI_Allreduce(&ok, &allOk, 1, MPI_INT, MPI_MIN, g.comm());
    if (!allOk) {
        err = "cannot write VTK output into '" + p.outdir + "'";
        return false;
    }
    return true;
}
