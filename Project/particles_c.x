#!/bin/sh
rm *.ppm *.jpg *.dmp *.sta
module load gnu
rm *.exe
gcc -pg -o particles_c.exe -O3 -fbounds-check particles_c.c -lm
./particles_c.exe
