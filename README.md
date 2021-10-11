# Parallel Computing project

This project implements our proposed solution for the assignment of the Parallel Computing course. 
More information on the project can be found on the [slides].

# How to compile
Clone the repo [here], ensure that CUDA compiler driver is installed and then run:
```sh
cd path/to/ParCom/Project
nvcc -arch=sm_70 -o particles.x particles_c.cu
```

## How to run
```sh
./particles.x
```

## Authors
This project was developed by [Chiara Marzano](https://github.com/ChiaraMarzano) and [Bruno Guindani](https://github.com/brunoguindani).

[here]: https://github.com/ChiaraMarzano/ParComProject
[slides]: https://github.com/ChiaraMarzano/ParComProject/blob/main/Project/ParCom%20Presentation.pdf
