CX 4220/CSE 6220 Introduction to High Performance Computing
Spring 2024
Programming Assignment 1

Instructions
1. Place the Makefile in the same directory as your C++ source files.
2. Open a terminal and navigate to the directory containing the Makefile.
3. Run 'make' to compile the program.
4. If you want to clean up the compiled files, run 'make clean'.
5. Use mpiexec or mpirun followed by the number of processes you want to run and the name of your executable. The general syntax is:

   mpiexec -n <number_of_processes> ./pi_calc <number of points to be used for the estimation>

   Or

   mpirun -n <number_of_processes> ./pi_calc <number of points to be used for the estimation.>
   Replace <number_of_processes> with the number of processes you want to run and <number of points to be used for the estimation> with the value of n.

Testing Platform
