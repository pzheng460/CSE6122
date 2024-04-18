CX 4220/CSE 6220 Introduction to High Performance Computing
Spring 2024
Programming Assignment 3

Instructions
1. Place the Makefile in the same directory as your C++ source files.
2. Open a terminal and navigate to the directory containing the Makefile.
3. Run 'make' to compile the program, which would generate 2 executables in this program.
4. If you want to clean up the compiled files, run 'make clean'.
5. Use mpiexec or mpirun followed by the number of processes you want to run and the name of your executable(In default, it is spmat). The general syntax is:

   $ mpiexec -np <number of processes> ./spmat <Dimension of the square matricese> <Sparsity parameter> <Print flag> <Output file name>

   Or

   $ mpirun -np <number of processes> ./spmat <Dimension of the square matricese> <Sparsity parameter> <Print flag> <Output file name>

   Replace <number_of_processes> with the number of processes you want to run and other parameters you want.

   Sample command line input:

   $ mpirun -np 8 ./spmat 10000 0.001 0 spmat_out

6. For the bonus part, makefile would generate an executable(In default, it is spmatBonus) at the same time. The general syntax is:

   $ mpiexec -np <number of processes> ./spmatBonus <Dimension of the square matricese> <Sparsity parameter> <Print flag> <Output file name1> <Output file name2>

   Or

   $ mpirun -np <number of processes> ./spmatBonus <Dimension of the square matricese> <Sparsity parameter> <Print flag> <Output file name1> <Output file name2>

   Replace <number_of_processes> with the number of processes you want to run and other parameters you want.

   Sample command line input:

   $ mpirun -np 4 ./spmatBonus 1000 0.01 1 spmat_out spmatBonus_out

Testing Platform
PACE-ICE cluster
