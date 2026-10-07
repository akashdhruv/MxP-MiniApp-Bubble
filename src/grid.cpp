// grid.cpp -- see grid.h.
//
// Spec: dev/tasks/init-project/spec.md, "Parallel contract":
//   "Halo exchange: 3-deep layers, MPI_Sendrecv in x then in y, so corner
//    guard cells arrive via the second sweep."
// and "Boundary conditions":
//   "Physical walls: zero-gradient (Neumann) extrapolation of phi into all 3
//    guard layers."
//
// The ordering matters.  The x sweep touches only the owned rows j in [0, ny);
// the y sweep then moves whole rows of width sx (guard columns included), so a
// corner guard cell is the neighbour's guard column value, which the x sweep
// had already made valid.  Neumann fill follows the same two-sweep order, so a
// corner that is physical in one direction and a rank boundary in the other
// comes out identical to the serial answer.

#include "grid.h"

#include <string>
#include <vector>

bool Grid::create(const Params& p, MPI_Comm comm, std::string& err) {
    destroy();

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(comm, &worldRank);
    MPI_Comm_size(comm, &worldSize);

    int dims[2] = {p.px, p.py};
    if (p.px > 0 && p.py > 0) {
        if (p.px * p.py != worldSize) {
            err = "px * py = " + std::to_string(p.px * p.py) + " does not match the " +
                  std::to_string(worldSize) + " MPI ranks";
            return false;
        }
    } else {
        // MPI_Dims_create honours any non-zero entry and fills the rest.
        MPI_Dims_create(worldSize, 2, dims);
    }

    if (p.nx % dims[0] != 0 || p.ny % dims[1] != 0) {
        err = "mesh " + std::to_string(p.nx) + "x" + std::to_string(p.ny) +
              " is not divisible by the process grid " + std::to_string(dims[0]) + "x" +
              std::to_string(dims[1]) + "; choose px/py or a different rank count";
        return false;
    }

    const int nxLocal = p.nx / dims[0];
    const int nyLocal = p.ny / dims[1];
    if (nxLocal < NGUARD || nyLocal < NGUARD) {
        err = "local block " + std::to_string(nxLocal) + "x" + std::to_string(nyLocal) +
              " is smaller than NGUARD = " + std::to_string(NGUARD) +
              "; use fewer ranks or a finer mesh";
        return false;
    }

    const int periods[2] = {0, 0};
    MPI_Cart_create(comm, 2, dims, periods, 0, &cart_);
    if (cart_ == MPI_COMM_NULL) {
        err = "MPI_Cart_create returned MPI_COMM_NULL";
        return false;
    }

    MPI_Comm_rank(cart_, &rank_);
    MPI_Comm_size(cart_, &nprocs_);
    MPI_Cart_coords(cart_, rank_, 2, coords_);
    MPI_Cart_shift(cart_, 0, 1, &nbrXlo_, &nbrXhi_);
    MPI_Cart_shift(cart_, 1, 1, &nbrYlo_, &nbrYhi_);

    dims_[0] = dims[0];
    dims_[1] = dims[1];
    ng_ = NGUARD;
    nxg_ = p.nx;
    nyg_ = p.ny;
    nx_ = nxLocal;
    ny_ = nyLocal;
    i0_ = coords_[0] * nxLocal;
    j0_ = coords_[1] * nyLocal;
    sx_ = nx_ + 2 * ng_;
    sy_ = ny_ + 2 * ng_;

    xmin_ = p.xmin;
    ymin_ = p.ymin;
    dx_ = (p.xmax - p.xmin) / static_cast<double>(p.nx);
    dy_ = (p.ymax - p.ymin) / static_cast<double>(p.ny);

    return true;
}

void Grid::destroy() {
    if (cart_ == MPI_COMM_NULL) return;
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) MPI_Comm_free(&cart_);
    cart_ = MPI_COMM_NULL;
}

Grid::~Grid() { destroy(); }

// x sweep: ng columns x ny owned rows, packed because columns are strided.
void Grid::exchangeX(std::vector<double>& f) const {
    const int count = ng_ * ny_;
    std::vector<double> sendLo(count), sendHi(count), recvLo(count), recvHi(count);

    for (int j = 0; j < ny_; ++j) {
        for (int k = 0; k < ng_; ++k) {
            sendLo[j * ng_ + k] = f[index(k, j)];                   // i = 0 .. ng-1
            sendHi[j * ng_ + k] = f[index(nx_ - ng_ + k, j)];       // i = nx-ng .. nx-1
        }
    }

    // to the low neighbour, from the high neighbour
    MPI_Sendrecv(sendLo.data(), count, MPI_DOUBLE, nbrXlo_, 100, recvHi.data(), count, MPI_DOUBLE,
                 nbrXhi_, 100, cart_, MPI_STATUS_IGNORE);
    // to the high neighbour, from the low neighbour
    MPI_Sendrecv(sendHi.data(), count, MPI_DOUBLE, nbrXhi_, 101, recvLo.data(), count, MPI_DOUBLE,
                 nbrXlo_, 101, cart_, MPI_STATUS_IGNORE);

    for (int j = 0; j < ny_; ++j) {
        for (int k = 0; k < ng_; ++k) {
            // low guard columns i = -ng .. -1
            f[index(-ng_ + k, j)] = atWallXlo() ? f[index(0, j)] : recvLo[j * ng_ + k];
            // high guard columns i = nx .. nx+ng-1
            f[index(nx_ + k, j)] = atWallXhi() ? f[index(nx_ - 1, j)] : recvHi[j * ng_ + k];
        }
    }
}

// y sweep: ng rows of the full width sx (guard columns included, so corners
// come across).  Rows are contiguous in this layout.
void Grid::exchangeY(std::vector<double>& f) const {
    const int count = ng_ * sx_;
    std::vector<double> recvLo(count), recvHi(count);

    const double* sendLo = &f[index(-ng_, 0)];           // rows j = 0 .. ng-1
    const double* sendHi = &f[index(-ng_, ny_ - ng_)];   // rows j = ny-ng .. ny-1

    MPI_Sendrecv(sendLo, count, MPI_DOUBLE, nbrYlo_, 200, recvHi.data(), count, MPI_DOUBLE, nbrYhi_,
                 200, cart_, MPI_STATUS_IGNORE);
    MPI_Sendrecv(sendHi, count, MPI_DOUBLE, nbrYhi_, 201, recvLo.data(), count, MPI_DOUBLE, nbrYlo_,
                 201, cart_, MPI_STATUS_IGNORE);

    for (int k = 0; k < ng_; ++k) {
        for (int i = -ng_; i < nx_ + ng_; ++i) {
            const int c = k * sx_ + (i + ng_);
            f[index(i, -ng_ + k)] = atWallYlo() ? f[index(i, 0)] : recvLo[c];
            f[index(i, ny_ + k)] = atWallYhi() ? f[index(i, ny_ - 1)] : recvHi[c];
        }
    }
}

void Grid::exchange(std::vector<double>& f) const {
    exchangeX(f);
    exchangeY(f);
}
