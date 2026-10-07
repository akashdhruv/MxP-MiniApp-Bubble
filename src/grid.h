// grid.h -- Grid: Cartesian decomposition, ownership map, guard cells,
//           halo exchange and physical boundary conditions.
//
// Spec: dev/tasks/init-project/spec.md, "Parallel contract" and
// "Boundary conditions".
//
// Storage convention
// ------------------
// A cell-centred field is a flat std::vector<double> of size() doubles laid
// out row-major in j:
//
//     index(i, j) = (j + ng) * sx + (i + ng)
//
// with the *owned* (interior) cells at i in [0, nx) and j in [0, ny) and the
// guard layers at i in [-ng, 0) U [nx, nx+ng) and likewise in j.  NGUARD = 3
// because the WENO5 stencil reaches i-3 .. i+3.
//
// MAC staggering (used from U2 on): dfun lives at cell centres, u on x-faces
// with u(i,j) the *low* x-face of cell (i,j), v on y-faces likewise.  The face
// arrays use the same index() mapping and the same allocation size, so a face
// field has room for the extra high face at i = nx.
#ifndef BUBBLE_GRID_H
#define BUBBLE_GRID_H

#include <mpi.h>

#include <cstddef>
#include <string>
#include <vector>

#include "params.h"

class Grid {
public:
    Grid() = default;
    ~Grid();

    // The grid owns an MPI communicator; copying it would double-free.
    Grid(const Grid&) = delete;
    Grid& operator=(const Grid&) = delete;

    // Build the Cartesian communicator (periods = 0,0) and the local ownership
    // map.  Dimensions come from MPI_Dims_create unless both px and py are set.
    // Returns false with `err` filled for a process grid that does not match
    // the rank count, a mesh the process grid does not divide, or a local block
    // too small to hold the guard layers.  Never truncates silently.
    bool create(const Params& p, MPI_Comm comm, std::string& err);

    // Free the Cartesian communicator (safe to call more than once).  Call it
    // before MPI_Finalize; the destructor also does it if MPI is still up.
    void destroy();

    // --- layout -------------------------------------------------------------
    int ng() const { return ng_; }
    int nx() const { return nx_; }          // owned cells in x
    int ny() const { return ny_; }          // owned cells in y
    int nxGlobal() const { return nxg_; }
    int nyGlobal() const { return nyg_; }
    int i0() const { return i0_; }          // global index of local i = 0
    int j0() const { return j0_; }
    int sx() const { return sx_; }          // nx + 2*ng
    int sy() const { return sy_; }          // ny + 2*ng
    std::size_t size() const { return static_cast<std::size_t>(sx_) * static_cast<std::size_t>(sy_); }

    int index(int i, int j) const { return (j + ng_) * sx_ + (i + ng_); }

    // --- geometry -----------------------------------------------------------
    double dx() const { return dx_; }
    double dy() const { return dy_; }
    double xCenter(int i) const { return xmin_ + (static_cast<double>(i0_ + i) + 0.5) * dx_; }
    double yCenter(int j) const { return ymin_ + (static_cast<double>(j0_ + j) + 0.5) * dy_; }
    double xFace(int i) const { return xmin_ + static_cast<double>(i0_ + i) * dx_; }
    double yFace(int j) const { return ymin_ + static_cast<double>(j0_ + j) * dy_; }

    // --- parallel environment -----------------------------------------------
    MPI_Comm comm() const { return cart_; }
    int rank() const { return rank_; }
    int nprocs() const { return nprocs_; }
    int dim(int k) const { return dims_[k]; }
    int coord(int k) const { return coords_[k]; }
    // MPI_PROC_NULL where the block touches a physical wall.
    int neighborXlo() const { return nbrXlo_; }
    int neighborXhi() const { return nbrXhi_; }
    int neighborYlo() const { return nbrYlo_; }
    int neighborYhi() const { return nbrYhi_; }
    bool atWallXlo() const { return nbrXlo_ == MPI_PROC_NULL; }
    bool atWallXhi() const { return nbrXhi_ == MPI_PROC_NULL; }
    bool atWallYlo() const { return nbrYlo_ == MPI_PROC_NULL; }
    bool atWallYhi() const { return nbrYhi_ == MPI_PROC_NULL; }

    std::vector<double> allocate(double value = 0.0) const {
        return std::vector<double>(size(), value);
    }

    // Fill all ng guard layers of a cell-centred field: MPI_Sendrecv in x,
    // then in y, so corner guard cells arrive on the second sweep; physical
    // walls get zero-gradient (Neumann) extrapolation in the same order.
    void exchange(std::vector<double>& f) const;

private:
    void exchangeX(std::vector<double>& f) const;
    void exchangeY(std::vector<double>& f) const;

    MPI_Comm cart_ = MPI_COMM_NULL;
    int rank_ = 0;
    int nprocs_ = 1;
    int dims_[2] = {1, 1};
    int coords_[2] = {0, 0};
    int nbrXlo_ = MPI_PROC_NULL;
    int nbrXhi_ = MPI_PROC_NULL;
    int nbrYlo_ = MPI_PROC_NULL;
    int nbrYhi_ = MPI_PROC_NULL;

    int ng_ = NGUARD;
    int nx_ = 0;
    int ny_ = 0;
    int nxg_ = 0;
    int nyg_ = 0;
    int i0_ = 0;
    int j0_ = 0;
    int sx_ = 0;
    int sy_ = 0;

    double xmin_ = 0.0;
    double ymin_ = 0.0;
    double dx_ = 0.0;
    double dy_ = 0.0;
};

#endif  // BUBBLE_GRID_H
