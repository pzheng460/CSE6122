#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>

int main(int argc, char *argv[]) {
    int rank, size, i;
    long long n, local_n, count = 0, total_count;
    double x, y, distance, pi_estimate, start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        if (argc != 2) {
            std::cerr << "Usage: " << argv[0] << " <number_of_points>\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        n = atoll(argv[1]);
        start_time = MPI_Wtime();
    }

    MPI_Bcast(&n, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    srand(time(NULL) + rank);

    local_n = n / size;

    for (i = 0; i < local_n; i++) {
        x = static_cast<double>(rand()) / RAND_MAX;
        y = static_cast<double>(rand()) / RAND_MAX;
        distance = sqrt(x * x + y * y);

        if (distance <= 1.0) {
            count++;
        }
    }

    MPI_Reduce(&count, &total_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        pi_estimate = 4.0 * total_count / n;
        end_time = MPI_Wtime();
        std::cout.precision(12);
        std::cout << std::fixed << pi_estimate << ", " << end_time - start_time << std::endl;
    }

    MPI_Finalize();
    return 0;
}

