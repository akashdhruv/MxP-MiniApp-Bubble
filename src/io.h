// io.h -- output writers: parallel VTK ImageData + history CSV (U5).
//
// Spec: dev/tasks/init-project/spec.md, "Output and post-processing".
//
//   out/dfun_<step>_<rank>.vti   one piece per rank, ASCII payload
//   out/dfun_<step>.pvti         master file written by rank 0, listing the
//                                pieces with their extents in *global* index
//                                space so the set reassembles to one field
//   out/history.csv              step, time, dt, area, mass, phi_min, phi_max,
//                                max_divergence, sign_flips
//
// ASCII at 128^2 is cheap and keeps the Python reader dependency-free (Spec,
// "Simplicity tradeoff").  All floating point is written with 17 significant
// digits so a -np 1 and a -np 4 run produce byte-identical payloads when the
// fields agree, which is what makes the rank-invariance check meaningful at
// file level too.
#ifndef BUBBLE_IO_H
#define BUBBLE_IO_H

#include <string>
#include <vector>

#include "grid.h"
#include "params.h"

// Integral diagnostics of one state, all MPI_Allreduce'd over the Cartesian
// communicator so every rank holds the same numbers.
struct Diagnostics {
    double area = 0.0;             // sum(H(phi)) dx dy, H smoothed over 1.5 cells
    double mass = 0.0;             // sum(H(phi) * phi) dx dy, the volume-weighted measure
    double phi_min = 0.0;
    double phi_max = 0.0;
    double max_divergence = 0.0;   // invariant 6
    long sign_flips = 0;           // redistance warning counter of the last step
};

// Smoothed Heaviside with half-width `eps` (Spec: 1.5 cells):
//   0                              phi < -eps
//   1                              phi >  eps
//   0.5 (1 + phi/eps + sin(pi phi/eps)/pi)   otherwise
double smoothedHeaviside(double phi, double eps);

// Half-width used for the area/mass integrals: 1.5 * max(dx, dy).
double heavisideWidth(const Grid& g);

// Area, mass and extrema of `dfun` plus max|div u| of the face velocity.
Diagnostics computeDiagnostics(const Grid& g, const std::vector<double>& dfun,
                               const std::vector<double>& u, const std::vector<double>& v,
                               long signFlips);

// Create `p.outdir` if it does not exist (rank 0 creates, all ranks wait).
// Returns false with `err` filled if it cannot be created.
bool ensureOutputDir(const Grid& g, const Params& p, std::string& err);

// Truncate out/history.csv and write the header row.  Collective; only rank 0
// touches the file.  Call once at the start of a run.
bool writeHistoryHeader(const Grid& g, const Params& p, std::string& err);

// Append one row to out/history.csv.  Collective; only rank 0 writes.
bool appendHistory(const Grid& g, const Params& p, int step, double time, double dt,
                   const Diagnostics& d, std::string& err);

// Write this rank's out/dfun_<step>_<rank>.vti and, on rank 0, the master
// out/dfun_<step>.pvti.  Cell data: dfun, and the face velocities averaged to
// cell centres as u, v (so the plotter can draw the field without knowing the
// MAC convention).  Collective: the piece extents are gathered for the .pvti.
bool writeFields(const Grid& g, const Params& p, int step, double time,
                 const std::vector<double>& dfun, const std::vector<double>& u,
                 const std::vector<double>& v, std::string& err);

// File names (relative to p.outdir), exposed so the tests and the tools agree
// with the writer.
std::string vtiName(int step, int rank);
std::string pvtiName(int step);

#endif  // BUBBLE_IO_H
