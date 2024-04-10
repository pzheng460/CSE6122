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

    if(argc != 5) {
        if(rank == 0) {
            std::cerr << "Invalid Input! Sample Input: $ mpirun -np 8 ./spmat 10000 0.001 0 spmat_out\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n = atoi(argv[1]);
    double sparsity = atof(argv[2]);
    int pf = atoi(argv[3]);
    string outputFile = argv[4];

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

    // Generate sparse matrices A, B and dense matrix C
    SparseMatrix sparseLocalA = generateSparseMatrix(n, sparsity, rank, size);
    SparseMatrix sparseLocalB = generateSparseMatrix(n, sparsity, rank, size);

    DenseMatrix denseLocalC(n * n / size, 0);

    // Synchronize and start counting time
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time, end_time;
    start_time = MPI_Wtime();

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

    // Get the left and right neighbors
    int left, right;
    MPI_Cart_shift(ring_comm, 0, 1, &left, &right);

    for (int step = 0; step < size; step++) {
        // Multiply sparseLocalA and sparseLocalB
        for (const auto& elemA : sparseLocalA) {
            int rowA = get<0>(elemA) % (n / size);
            int colA = get<1>(elemA);
            uint64_t valueA = get<2>(elemA);
            for (const auto& elemB : TransposedSparseLocalB) {
                int rowB = get<1>(elemB);
                int colB = get<0>(elemB);
                uint64_t valueB = get<2>(elemB);
                if (colA == rowB) {
                    denseLocalC[rowA * n + colB] += valueA * valueB;
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
    end_time = MPI_Wtime();

    if (pf == 1) {
        DenseMatrix denseA(n * n, 0);
        DenseMatrix denseB(n * n, 0);
        DenseMatrix denseC(n * n, 0);

        DenseMatrix denseLocalA = convertToDenseMatrix(sparseLocalA, n, size);
        DenseMatrix denseLocalB = convertToDenseMatrix(sparseLocalB, n, size);

        // Gather the dense matrices A, B and C to rank 0
        MPI_Gather(denseLocalA.data(), n * n / size, MPI_UINT64_T, denseA.data(), n * n / size, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        MPI_Gather(denseLocalB.data(), n * n / size, MPI_UINT64_T, denseB.data(), n * n / size, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        MPI_Gather(denseLocalC.data(), n * n / size, MPI_UINT64_T, denseC.data(), n * n / size, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            ofstream outStream(outputFile, ios::out);
            if(outStream.is_open()) {
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

    if (rank == 0) {
        // The time taken in milliseconds (round to 6 decimal places) must be printed to the terminal using printf.
        double time = (end_time - start_time) * 1000;
        printf("%.6f milliseconds\n", time);
    }

    MPI_Finalize();
    return 0;
}
