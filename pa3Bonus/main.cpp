#include <mpi.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <chrono>
#include <algorithm>
#include <unordered_map>

using namespace std;

using SparseElement = tuple<int, int, uint64_t>;
using SparseMatrix = vector<SparseElement>;
using DenseMatrix = vector<uint64_t>;

// Generate a sparse matrix
SparseMatrix generateSparseMatrix(int n, double sparsity, int rank, int size) {
    SparseMatrix matrix;

    auto now = chrono::high_resolution_clock::now();
    auto seed = now.time_since_epoch().count() + rank;
    mt19937_64 engine(seed); // Random number engine
    uniform_real_distribution<double> distDouble(0.0, 1.0);
    uint64_t root = static_cast<uint64_t>(sqrt(UINT64_MAX / n));
    uniform_int_distribution<uint64_t> distValue(0, root);
//    uniform_int_distribution<uint64_t> distValue(0, 10);

    int rowsPerProc = n / size;
    int startRow = rank * rowsPerProc;
    int endRow = startRow + rowsPerProc;

    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < n; ++j) {
            if (distDouble(engine) < sparsity) {
                uint64_t value = distValue(engine) + 1; // Avoid zero
                matrix.push_back(make_tuple(i, j, value));
            }
        }
    }
    return matrix;
}

// Generate a sparse matrix in 2D-partitioning
SparseMatrix generateSparseMatrix2D(DenseMatrix Matrix1D, int n, int rank, int size, int dim) {
    SparseMatrix matrix2D;

    int blockSize = n / dim;

    int colStart = (rank / dim) * blockSize;
    int colEnd = colStart + blockSize;
    int rowStart = (rank % dim) * blockSize;
    int rowEnd = rowStart + blockSize;

    for (int i = rowStart; i < rowEnd; i++) {
        for (int j = colStart; j < colEnd; j++) {
            uint64_t value = Matrix1D[i * n + j];
            if (value != 0) {
                matrix2D.push_back(make_tuple(i, j, value));
            }
        }
    }
    return matrix2D;
}

DenseMatrix rearrangeDenseMatrix2D(DenseMatrix Matrix2D, int n, int size, int dim) {
    DenseMatrix rearrangeMatrix(n * n, 0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int row = i / (n / dim);
            int col = j / (n / dim);
            int proc = col * dim + row;
            int localRow = i % (n / dim);
            int localCol = j % (n / dim);
            rearrangeMatrix[i * n + j] = Matrix2D[proc * n * n / size + localRow * n / dim + localCol];
        }
    }

    return rearrangeMatrix;
}

// Transpose a sparse matrix
SparseMatrix transposeSparseMatrix(const SparseMatrix& localB, int n, int rank, int size, MPI_Comm comm) {
    // Store the number of elements to be sent to each process or received from each process
    vector<int> sendCounts(size, 0);
    vector<int> recvCounts(size, 0);
    for (const auto& elem : localB) {
        int targetProc = get<1>(elem) * size / n; // Calculate the target process based on the column number
        sendCounts[targetProc]++;
    }

    // Exchange the sendCounts to get the recvCounts
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, comm);

    // Calculate the displacements
    vector<int> sdispls(size, 0), rdispls(size, 0);
    for (int i = 1; i < size; i++) {
        sdispls[i] = sdispls[i - 1] + sendCounts[i - 1];
        rdispls[i] = rdispls[i - 1] + recvCounts[i - 1];
    }

    // Prepare the send buffer
    vector<SparseElement> sendBuffer(localB.size());

    // copy sdispls
    vector<int> sdispls_copy = sdispls;

    // Fill the send buffer
    for (const auto& elem : localB) {
        int targetProc = get<1>(elem) * size / n;
        int index = sdispls_copy[targetProc]++;
        sendBuffer[index] = elem;
    }

    // Calculate the total number of elements to be received
    int totalRecv = accumulate(recvCounts.begin(), recvCounts.end(), 0);
    vector<SparseElement> recvBuffer(totalRecv);

    // Convert the counts and displacements to bytes
    for (int i = 0; i < size; i++) {
        sendCounts[i] *= sizeof(SparseElement);
        recvCounts[i] *= sizeof(SparseElement);
        sdispls[i] *= sizeof(SparseElement);
        rdispls[i] *= sizeof(SparseElement);
    }

    // Exchange the data
    MPI_Alltoallv(sendBuffer.data(), sendCounts.data(), sdispls.data(), MPI_BYTE,
                  recvBuffer.data(), recvCounts.data(), rdispls.data(), MPI_BYTE, comm);

    // Transpose the received data
    SparseMatrix transposedLocalB;
    for (const auto& elem : recvBuffer) {
        transposedLocalB.push_back(make_tuple(get<1>(elem), get<0>(elem), get<2>(elem)));
    }

    return transposedLocalB;
}

// Convert a sparse matrix to a dense matrix
DenseMatrix convertToDenseMatrix(const SparseMatrix& sparseMatrix, int n, int size) {
    DenseMatrix denseMatrix(n * n / size, 0);
    for (const auto& elem : sparseMatrix) {
        int row = get<0>(elem) % (n / size);
        int col = get<1>(elem);
        uint64_t value = get<2>(elem);
        denseMatrix[row * n + col] = value;
    }
    return denseMatrix;
}

DenseMatrix convertToDenseMatrix2D(const SparseMatrix& sparseMatrix, int n, int size, int dim) {
    DenseMatrix denseMatrix(n * n / size, 0);
    for (const auto& elem : sparseMatrix) {
        int row = get<0>(elem) % (n / dim);
        int col = get<1>(elem) % (n / dim);
        uint64_t value = get<2>(elem);

        denseMatrix[row * n / dim + col] = value;
    }
    return denseMatrix;
}

// Print a dense matrix to a file stream
void printDenseMatrix(std::ofstream& outStream, const DenseMatrix& matrix, int n) {
    for (size_t i = 0; i < matrix.size(); i++) {
        outStream << matrix[i];
        if ((i + 1) % n == 0) {
            if (i + 1 < matrix.size())
                outStream << endl;
        } else {
            outStream << " ";
        }
    }
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(argc != 6) {
        if(rank == 0) {
            std::cerr << "Invalid Input! Sample Input: $ mpirun -np 8 ./spmat 10000 0.001 0 spmat_out\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int dim = static_cast<int>(sqrt(size));
    if (dim * dim != size) {
        if (rank == 0) {
            cerr << "The number of processes must be a perfect square." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n = atoi(argv[1]);
    double sparsity = atof(argv[2]);
    int pf = atoi(argv[3]);
    string outputFile0 = argv[4];
    string outputFile1 = argv[5];

    if (sparsity <= 0 || sparsity > 1) {
        if (rank == 0) {
            std::cerr << "Invalid sparsity value! Sparsity must be between 0 and 1." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (pf != 0 && pf != 1) {
        if (rank == 0) {
            std::cerr << "Invalid print flag! Print flag must be 0 or 1." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    DenseMatrix denseA(n * n, 0);
    DenseMatrix denseB(n * n, 0);
    DenseMatrix denseC(n * n, 0);

    // Generate sparse matrices A, B and dense matrix C
    SparseMatrix sparseLocalA = generateSparseMatrix(n, sparsity, rank, size);
    SparseMatrix sparseLocalB = generateSparseMatrix(n, sparsity, rank, size);

    DenseMatrix denseLocalC(n * n / size, 0);

    // Synchronize and start counting time
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time0, end_time0;
    start_time0 = MPI_Wtime();

    // Transpose matrix B
    SparseMatrix TransposedSparseLocalB = transposeSparseMatrix(sparseLocalB, n, rank, size, MPI_COMM_WORLD);

    // Create a ring topology
    int dimensions = 1;
    int dimension_size[1] = {size}; // size is the number of processors
    int periods[1] = {1}; // 1 means the topology is periodic
    MPI_Comm ring_comm;
    MPI_Cart_create(MPI_COMM_WORLD, dimensions, dimension_size, periods, 1, &ring_comm);

    int ring_rank, ring_size;
    MPI_Comm_rank(ring_comm, &ring_rank); // Ring rank
    MPI_Comm_size(ring_comm, &ring_size); // Ring size, should be same as size

    // Compute the left and right neighbors
    int left, right;
    MPI_Cart_shift(ring_comm, 0, 1, &left, &right);

    sort(sparseLocalA.begin(), sparseLocalA.end(), [](const auto& a, const auto& b) {
        return get<1>(a) < get<1>(b);
    });

    for (int step = 0; step < size; step++) {
        // Multiply sparseLocalA and sparseLocalB
//        for (const auto& elemA : sparseLocalA) {
//            int rowA = get<0>(elemA) % (n / size);
//            int colA = get<1>(elemA);
//            uint64_t valueA = get<2>(elemA);
//            for (const auto& elemB : TransposedSparseLocalB) {
//                int rowB = get<1>(elemB);
//                int colB = get<0>(elemB);
//                uint64_t valueB = get<2>(elemB);
//                if (colA == rowB) {
//                    denseLocalC[rowA * n + colB] += valueA * valueB;
//                }
//            }
//        }
        vector<SparseMatrix> sparseLocalGroupA(n / size);
        vector<SparseMatrix> TransposedsparseLocalGroupB(n / size);

        for (const auto& elemA : sparseLocalA) {
            int index = get<0>(elemA) % (n / size);
            sparseLocalGroupA[index].push_back(elemA);
        }
        for (const auto& elemB : TransposedSparseLocalB) {
            int index = get<0>(elemB) % (n / size);
            TransposedsparseLocalGroupB[index].push_back(elemB);
        }

        for (int iA = 0; iA < n / size; iA++) {
            for (int iB = 0; iB < n / size; iB++) {
                long unsigned int jA = 0, jB = 0;
                while (jA < sparseLocalGroupA[iA].size() && jB < TransposedsparseLocalGroupB[iB].size()) {
                    const auto& elemA = sparseLocalGroupA[iA][jA];
                    const auto& elemB = TransposedsparseLocalGroupB[iB][jB];

                    int rowA = get<0>(elemA) % (n / size);
                    int colA = get<1>(elemA);
                    uint64_t valueA = get<2>(elemA);

                    int rowB = get<1>(elemB);
                    int colB = get<0>(elemB);
                    uint64_t valueB = get<2>(elemB);

                    if (colA == rowB) {
                        denseLocalC[rowA * n + colB] += valueA * valueB;
                        jA++;
                        jB++;
                    } else if (colA < rowB) {
                        jA++;
                    } else {
                        jB++;
                    }
                }
            }
        }

        // Shift sparseLocalB to the left
        vector<SparseElement> sendBuffer = TransposedSparseLocalB;
        vector<SparseElement> recvBuffer;

        int send_size = sendBuffer.size();
        int recv_size;

        // Exchange data size
        MPI_Sendrecv(&send_size, 1, MPI_INT, left, 0, &recv_size, 1, MPI_INT, right, 0, ring_comm, MPI_STATUS_IGNORE);

        recvBuffer.resize(recv_size);

        // Exchange data
        MPI_Sendrecv(sendBuffer.data(), sendBuffer.size() * sizeof(SparseElement), MPI_BYTE, left, 0,
                     recvBuffer.data(), recvBuffer.size() * sizeof(SparseElement), MPI_BYTE, right, 0, ring_comm, MPI_STATUS_IGNORE);
        TransposedSparseLocalB = recvBuffer;
    }

    // Synchronize all processes and stop counting time
    MPI_Barrier(MPI_COMM_WORLD);
    end_time0 = MPI_Wtime();

    DenseMatrix denseLocalA = convertToDenseMatrix(sparseLocalA, n, size);
    DenseMatrix denseLocalB = convertToDenseMatrix(sparseLocalB, n, size);

    if (pf == 1) {
        denseLocalA = convertToDenseMatrix(sparseLocalA, n, size);
        denseLocalB = convertToDenseMatrix(sparseLocalB, n, size);

        // Gather the dense matrices A, B and C to rank 0
        MPI_Gather(denseLocalA.data(), n * n / size, MPI_UINT64_T, denseA.data(), n * n / size, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        MPI_Gather(denseLocalB.data(), n * n / size, MPI_UINT64_T, denseB.data(), n * n / size, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        MPI_Gather(denseLocalC.data(), n * n / size, MPI_UINT64_T, denseC.data(), n * n / size, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            ofstream outStream(outputFile0, ios::out);
            if (outStream.is_open()) {
                // Write matrices A, B, and C to outStream
                printDenseMatrix(outStream, denseA, n);
                outStream << endl;
                outStream << endl;

                printDenseMatrix(outStream, denseB, n);
                outStream << endl;
                outStream << endl;

                printDenseMatrix(outStream, denseC, n);
            }
            outStream.close();
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allgather(denseLocalA.data(), n * n / size, MPI_UINT64_T,
                  denseA.data(), n * n / size, MPI_UINT64_T,
                  MPI_COMM_WORLD);

    MPI_Allgather(denseLocalB.data(), n * n / size, MPI_UINT64_T,
                  denseB.data(), n * n / size, MPI_UINT64_T,
                  MPI_COMM_WORLD);

    SparseMatrix sparseLocalA2D = generateSparseMatrix2D(denseA, n, rank, size, dim);
    SparseMatrix sparseLocalB2D = generateSparseMatrix2D(denseB, n, rank, size, dim);

    // reinitialize the denseLocalC
    denseLocalC.assign(n * n / size, 0);

    // Synchronize and start counting time
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time1, end_time1;
    start_time1 = MPI_Wtime();

    // Create a 2-D mesh with wraparound link (2-D torus) topology
    int dimension_sizes[2] = {dim, dim};
    int periods2D[2] = {1, 1};
    int coords[2];
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimension_sizes, periods2D, 1, &grid_comm);

    int grid_rank, grid_size;
    MPI_Comm_rank(grid_comm, &grid_rank); // Grid rank
    MPI_Comm_size(grid_comm, &grid_size); // Grid size
    MPI_Cart_coords(grid_comm, grid_rank, 2, coords); // Get the coordinates of the process

    int src[2], dst[2];

    MPI_Cart_shift(grid_comm, 0, -1, &src[0], &dst[0]);
    MPI_Cart_shift(grid_comm, 1, -1, &src[1], &dst[1]);

    for (int step = 0; step < dim; step++) {
        if (coords[1] > step) {
            // Shift sparseLocalA to the left
            vector<SparseElement> sendBuffer0 = sparseLocalA2D;
            vector<SparseElement> recvBuffer0;

            int send_size0 = sendBuffer0.size();
            int recv_size0;

            // Exchange data size
            MPI_Sendrecv(&send_size0, 1, MPI_INT, dst[0], 0, &recv_size0, 1, MPI_INT, src[0], 0, grid_comm, MPI_STATUS_IGNORE);

            recvBuffer0.resize(recv_size0);

            // Exchange data
            MPI_Sendrecv(sendBuffer0.data(), sendBuffer0.size() * sizeof(SparseElement), MPI_BYTE, dst[0], 0,
                         recvBuffer0.data(), recvBuffer0.size() * sizeof(SparseElement), MPI_BYTE, src[0], 0, grid_comm, MPI_STATUS_IGNORE);
            sparseLocalA2D = recvBuffer0;
        }

        if (coords[0] > step) {
            // Shift sparseLocalB up
            vector<SparseElement> sendBuffer1 = sparseLocalB2D;
            vector<SparseElement> recvBuffer1;

            int send_size1 = sendBuffer1.size();
            int recv_size1;

            // Exchange data size
            MPI_Sendrecv(&send_size1, 1, MPI_INT, dst[1], 0, &recv_size1, 1, MPI_INT, src[1], 0, grid_comm, MPI_STATUS_IGNORE);

            recvBuffer1.resize(recv_size1);

            // Exchange data
            MPI_Sendrecv(sendBuffer1.data(), sendBuffer1.size() * sizeof(SparseElement), MPI_BYTE, dst[1], 0,
                         recvBuffer1.data(), recvBuffer1.size() * sizeof(SparseElement), MPI_BYTE, src[1], 0, grid_comm, MPI_STATUS_IGNORE);
            sparseLocalB2D = recvBuffer1;
        }
    }

//    sort(sparseLocalA2D.begin(), sparseLocalA2D.end(), [](const auto& a, const auto& b) {
//        return get<1>(a) < get<1>(b);
//    });
//
//    sort(sparseLocalB2D.begin(), sparseLocalB2D.end(), [](const auto& a, const auto& b) {
//        return get<0>(a) < get<0>(b);
//    });

    for (int step = 0; step < dim; step++) {
        // Multiply sparseLocalA and sparseLocalB
        unordered_map<uint64_t, SparseMatrix> sparseLocalMapB2D;
        for (const auto& elemB : sparseLocalB2D) {
            int index = get<0>(elemB) % (n / dim);
            sparseLocalMapB2D[index].push_back(elemB);
        }

        for (const auto& elemA : sparseLocalA2D) {
            int rowA = get<0>(elemA) % (n / dim);
            int colA = get<1>(elemA) % (n / dim);
            uint64_t valueA = get<2>(elemA);
            if (sparseLocalMapB2D.find(colA) != sparseLocalMapB2D.end()) {
                for (const auto& elemB : sparseLocalMapB2D[colA]) {
                    int colB = get<1>(elemB) % (n / dim);
                    uint64_t valueB = get<2>(elemB);
                    denseLocalC[rowA * n / dim + colB] += valueA * valueB;
                }
            }
        }

//        for (const auto& elemA : sparseLocalA2D) {
//            int rowA = get<0>(elemA) % (n / dim);
//            int colA = get<1>(elemA) % (n / dim);
//            uint64_t valueA = get<2>(elemA);
//            for (const auto& elemB : sparseLocalB2D) {
//                int rowB = get<0>(elemB) % (n / dim);
//                int colB = get<1>(elemB) % (n / dim);
//                uint64_t valueB = get<2>(elemB);
//                if (colA == rowB) {
//                    denseLocalC[rowA * n / dim + colB] += valueA * valueB;
//                }
//            }
//        }

//        vector<SparseMatrix> sparseLocalGroupA2D(n / dim);
//        vector<SparseMatrix> sparseLocalGroupB2D(n / dim);
//
//        for (const auto& elemA : sparseLocalA2D) {
//            int index = get<0>(elemA) % (n / dim);
//            sparseLocalGroupA2D[index].push_back(elemA);
//        }
//        for (const auto& elemB : sparseLocalB2D) {
//            int index = get<1>(elemB) % (n / dim);
//            sparseLocalGroupB2D[index].push_back(elemB);
//        }
//
//        for (int iA = 0; iA < n / dim; iA++) {
//            for (int iB = 0; iB < n / dim; iB++) {
//                long unsigned int jA = 0, jB = 0;
//                while (jA < sparseLocalGroupA2D[iA].size() && jB < sparseLocalGroupB2D[iB].size()) {
//                    const auto& elemA = sparseLocalGroupA2D[iA][jA];
//                    const auto& elemB = sparseLocalGroupB2D[iB][jB];
//
//                    int rowA = get<0>(elemA) % (n / dim);
//                    int colA = get<1>(elemA) % (n / dim);
//                    uint64_t valueA = get<2>(elemA);
//
//                    int rowB = get<0>(elemB) % (n / dim);
//                    int colB = get<1>(elemB) % (n / dim);
//                    uint64_t valueB = get<2>(elemB);
//
//                    if (colA == rowB) {
//                        denseLocalC[rowA * n / dim + colB] += valueA * valueB;
//                        jA++;
//                        jB++;
//                    } else if (colA < rowB) {
//                        jA++;
//                    } else {
//                        jB++;
//                    }
//                }
//            }
//        }

        // Shift sparseLocalA to the left
        vector<SparseElement> sendBuffer0 = sparseLocalA2D;
        vector<SparseElement> recvBuffer0;

        int send_size0 = sendBuffer0.size();
        int recv_size0;

        // Exchange data size
        MPI_Sendrecv(&send_size0, 1, MPI_INT, dst[0], 0, &recv_size0, 1, MPI_INT, src[0], 0, grid_comm, MPI_STATUS_IGNORE);

        recvBuffer0.resize(recv_size0);

        // Exchange data
        MPI_Sendrecv(sendBuffer0.data(), sendBuffer0.size() * sizeof(SparseElement), MPI_BYTE, dst[0], 0,
                     recvBuffer0.data(), recvBuffer0.size() * sizeof(SparseElement), MPI_BYTE, src[0], 0, grid_comm, MPI_STATUS_IGNORE);
        sparseLocalA2D = recvBuffer0;

        // Shift sparseLocalB up
        vector<SparseElement> sendBuffer1 = sparseLocalB2D;
        vector<SparseElement> recvBuffer1;

        int send_size1 = sendBuffer1.size();
        int recv_size1;

        // Exchange data size
        MPI_Sendrecv(&send_size1, 1, MPI_INT, dst[1], 0, &recv_size1, 1, MPI_INT, src[1], 0, grid_comm, MPI_STATUS_IGNORE);

        recvBuffer1.resize(recv_size1);

        // Exchange data
        MPI_Sendrecv(sendBuffer1.data(), sendBuffer1.size() * sizeof(SparseElement), MPI_BYTE, dst[1], 0,
                     recvBuffer1.data(), recvBuffer1.size() * sizeof(SparseElement), MPI_BYTE, src[1], 0, grid_comm, MPI_STATUS_IGNORE);
        sparseLocalB2D = recvBuffer1;
    }

    // Synchronize all processes and stop counting time
    MPI_Barrier(MPI_COMM_WORLD);
    end_time1 = MPI_Wtime();

    if (pf == 1) {
        MPI_Gather(denseLocalC.data(), n * n / size, MPI_UINT64_T, denseC.data(), n * n / size, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        // Rearrange the dense matrix C
        if (rank == 0) {
            DenseMatrix rearrangedDenseC = rearrangeDenseMatrix2D(denseC, n, size, dim);

            ofstream outStream(outputFile1, ios::out);
            if (outStream.is_open()) {
                // Write matrices A, B, and C to outStream
                printDenseMatrix(outStream, denseA, n);
                outStream << endl;
                outStream << endl;

                printDenseMatrix(outStream, denseB, n);
                outStream << endl;
                outStream << endl;

                printDenseMatrix(outStream, rearrangedDenseC, n);
            }
            outStream.close();
        }
    }

    if (rank == 0) {
        // The time taken in milliseconds (round to 6 decimal places) must be printed to the terminal using printf.
        double time0 = (end_time0 - start_time0) * 1000;
        printf("Spmat: %.6f milliseconds\n", time0);
        double time1 = (end_time1 - start_time1) * 1000;
        printf("SpmatBonus: %.6f milliseconds\n", time1);
    }


    MPI_Finalize();
    return 0;
}
