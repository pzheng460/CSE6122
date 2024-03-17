CX 4220/CSE 6220 Introduction to High Performance Computing
Spring 2024
Programming Assignment 2

Instructions
1. Place the Makefile in the same directory as your C++ source files.
2. Open a terminal and navigate to the directory containing the Makefile.
3. Run 'make' to compile the program.
4. If you want to clean up the compiled files, run 'make clean'.
5. Use mpiexec or mpirun followed by the number of processes you want to run and the name of your executable(In default, it is transpose). The general syntax is:

   $ mpiexec -np <number of processes> ./transpose <input file> <output file> <algorithm> <matrix size>

   Or

   $ mpirun -np <number of processes> ./transpose <input file> <output file> <algorithm> <matrix size>

   Replace <number_of_processes> with the number of processes you want to run and other parameters you want.

   Sample command line input:

   $ mpirun -np 8 ./transpose matrix.txt transpose.txt m 24

Testing Platform
PACE-ICE cluster
