rm -f particles_c_mpi
mpiicc -O3 -qopenmp -qopt-report-phase=openmp particles_c_mpi.c -o particles_c_mpi
rm -f *.dppm *.ppm *.dmp *.sta
export OMP_NUM_THREADS=2
mpirun -np 2 ./particles_c_mpi
