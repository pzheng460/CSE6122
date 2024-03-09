#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>

using namespace std;

// Hypercubic permutation algorithm
void HPC_Alltoall_H(int *sendbuf, int sendcount, MPI_Datatype sendtype,
                    int *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
    // To be implemented
}
// Arbitrary permutation algorithm
void HPC_Alltoall_A(int *sendbuf, int sendcount, MPI_Datatype sendtype,
                    int *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
    // To be implemented
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 5) {
        if (rank == 0) {
            cerr << "Invalid Input! Sample Input: mpirun -np 8 ./transpose matrix.txt transpose.txt a 24" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Parse command line arguments
    string inputFile = argv[1];
    string outputFile = argv[2];
    char algorithm = argv[3][0];
    int n = atoi(argv[4]);

    // Make sure the algorithm is valid
    if (algorithm != 'h' && algorithm != 'a' && algorithm != 'm') {
        if (rank == 0) {
            cerr << "Invalid algorithm!" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    vector<int> matrix;
    // Read file
    if (rank == 0) {
        matrix.resize(n * n);
        ifstream file(inputFile);
        if (!file.is_open()) {
            cerr << "Can not open the file: " << inputFile << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < n * n; i++) {
            file >> matrix[i];
        }
        file.close();
    }


    vector<int> transposed(n * n);   // matrix to store the final transposed matrix
    vector<int> local_matrix(n * n / size);
    vector<int> local_transposed(n * n / size);

    // Scatter the matrix rows to all processors
    MPI_Scatter(matrix.data(), n * n / size, MPI_INT, local_matrix.data(), n * n / size, MPI_INT, 0, MPI_COMM_WORLD);

    // Synchronize and start counting time
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time, end_time;
    start_time = MPI_Wtime();

    // Transpose the local matrix in different blocks and then transpose the matrix to make the data contiguous
    for (int i = 0; i < n / size; i++) {
        for (int j = 0; j < n; j++) {
            int y = j - j % (n / size);
            local_transposed[(i + y) * (n / size) + j - y] = local_matrix[i * n + j];
        }
    }

    // Alltoall based on the option
    if (algorithm == 'h') {
        HPC_Alltoall_H(local_transposed.data(), (n / size) * (n / size), MPI_INT, local_matrix.data(), (n / size) * (n / size), MPI_INT, MPI_COMM_WORLD);
    } else if (algorithm == 'a') {
        HPC_Alltoall_A(local_transposed.data(), (n / size) * (n / size), MPI_INT, local_matrix.data(), (n / size) * (n / size), MPI_INT, MPI_COMM_WORLD);
    } else if (algorithm == 'm') {
        MPI_Alltoall(local_transposed.data(), (n / size) * (n / size), MPI_INT, local_matrix.data(), (n / size) * (n / size), MPI_INT, MPI_COMM_WORLD);
    }

    // Transpose the matrix to original form
    for (int i = 0; i < n / size; i++) {
        for (int j = 0; j < n; j++) {
            local_transposed[i * n + j] = local_matrix[j * (n / size) + i];
        }
    }

    // Synchronize all processes and stop counting time
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    // Gather the transposed matrix from all processors
    MPI_Gather(local_transposed.data(), n * n / size, MPI_INT, transposed.data(), n * n / size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Write file
        ofstream outFile(outputFile);
        if (!outFile.is_open()) {
            cerr << "Can not open the file!"<< endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                outFile << transposed[i * n + j] << " ";
            }
            outFile << "\n";
        }
        outFile.close();

        // The time taken in milliseconds (round to 6 decimal places) must be printed to the terminal using printf.
        double time = (end_time - start_time) * 1000;
        printf("%.6f milliseconds\n", time);
    }

    MPI_Finalize();
    return 0;
}
