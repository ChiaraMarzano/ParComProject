/*
!                          Program Particles
!  mimics the behaviour of a system of particles affected by mutual forces
!
!  Final application of the Course "Parallel Computing using MPI and OpenMP"
!
!  This program is meant to be used by course participants for demonstrating
!  the abilities acquired in optimizing and parallelising programs.
!
!  The techniques learnt at the course should be extensively used in order to
!  improve the program response times as much as possible, while gaining
!  the same or very closed results, i.e. the series of final produced images
!  and statistical results.
!
!  The code implemented herewith has been written for course exercise only,
!  therefore source code, algorithms and produced results must not be
!  trusted nor used for anything different from their original purpose.
!
!  Description of the program:
!  a squared grid is hit by a field whose result is the distribution of particles
!  with different properties.
!  After having been created the particles move under the effect of mutual
!  forces.
!
!  Would you please send comments to m.cremonesi@cineca.it
!
!  Program outline:
!  1 - the program starts reading the initial values (InitGrid)
!  2 - the generating field is computed (GeneratingField)
!  3 - the set of created particles is computed (ParticleGeneration)
!  4 - the evolution of the system of particles is computed (SystemEvolution)
!
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define SHARED_MEM_MAX_THREADS 1024
#define index2D(i, j, LD1) i + ((j)*LD1)  // element position in 2-D arrays

int deviceId, num_SMs;  // device information
int MaxIters, MaxSteps;  // parameters
double TimeBit;   // evolution time steps

struct i2dGrid {
    int EX, EY; // extensions in X and Y directions
    double Xs, Xe, Ys, Ye; // initial and final value for X and Y directions
    int *Values; // 2D matrix of values
} GenFieldGrid, ParticleGrid;

// Device grid
i2dGrid *GenFieldGrid_dev;

void print_i2dGrid(struct i2dGrid g) {
    printf("i2dGrid: EX, EY = %d, %d\n", g.EX, g.EY);
    printf("         Xs, Xe = %lf, %lf; Ys, Ye = %lf, %lf\n", g.Xs, g.Xe, g.Ys, g.Ye);
}

struct particle {
    double weight, x, y, vx, vy, fx, fy;
};

void print_particle(struct particle p) {
    printf("particle: weight=%lf, x,y=(%lf,%lf), vx,vy=(%lf,%lf), fx,fy=(%lf,%lf)\n",
           p.weight, p.x, p.y, p.vx, p.vy, p.fx, p.fy);
}

struct Population {
    int np;
    double *weight, *x, *y, *vx, *vy; // particles have a position and few other properties
} Particles;

__host__ __device__ void print_Population(struct Population p) {
    printf("Population: np = %d\n", p.np);
}



void DumpPopulation(struct Population p, int t) {
    /*
     * save population values on file
    */
    char fname[80];
    FILE *dump;

    sprintf(fname, "Population%4.4d.dmp\0", t);
    dump = fopen(fname, "w");
    if (dump == NULL) {
        fprintf(stderr, "Error write open file %s\n", fname);
        exit(1);
    }
    fwrite(&p.np, sizeof((int) 1), 1, dump);
    fwrite(p.weight, sizeof((double) 1.0), p.np, dump);
    fwrite(p.x, sizeof((double) 1.0), p.np, dump);
    fwrite(p.y, sizeof((double) 1.0), p.np, dump);
    fclose(dump);
}



__global__ void ParallelComputeStats(struct Population *pop, double *returns)
{
    if (threadIdx.x >= blockDim.x) {
        return;
    }

    // Declare shared memory arrays
    __shared__ double local_wmins[SHARED_MEM_MAX_THREADS];
    __shared__ double local_wmaxs[SHARED_MEM_MAX_THREADS];
    __shared__ double local_wsums[SHARED_MEM_MAX_THREADS];
    __shared__ double local_xgs[SHARED_MEM_MAX_THREADS];
    __shared__ double local_ygs[SHARED_MEM_MAX_THREADS];

    // Initialize size of data chunk for this thread, while adjusting for case
    // of non-exact division
    int total_size = pop->np;
    int local_size = total_size / blockDim.x;
    int remainder = total_size % blockDim.x;
    if (remainder != 0 && threadIdx.x < remainder)
        local_size++;
    // Initialize start and end data indexes for this thread, while adjusting
    // for case of non-exact division
    int first_val_idx = threadIdx.x * local_size;
    if (threadIdx.x >= remainder)
        first_val_idx += remainder;
    int last_val_idx = first_val_idx + local_size;

    // Initialize current local values
    double local_wmin_curr = pop->weight[first_val_idx];
    double local_wmax_curr = pop->weight[first_val_idx];
    double local_wsum_curr = pop->weight[first_val_idx];
    double local_xg_curr = pop->weight[first_val_idx] * pop->x[first_val_idx];
    double local_yg_curr = pop->weight[first_val_idx] * pop->y[first_val_idx];
    double w;
    int i;

    // Loop to compute local values
    for (i = first_val_idx+1; i < last_val_idx; i++) {
        w = pop->weight[i];
        // Update sums
        local_wsum_curr += w;
        local_xg_curr += w * pop->x[i];
        local_yg_curr += w * pop->y[i];
        // Update optima
        if (local_wmin_curr > w)
            local_wmin_curr = w;
        if (local_wmax_curr < w)
            local_wmax_curr = w;
    }
    // Assign values to local arrays
    local_wmins[threadIdx.x] = local_wmin_curr;
    local_wmaxs[threadIdx.x] = local_wmax_curr;
    local_wsums[threadIdx.x] = local_wsum_curr;
    local_xgs[threadIdx.x] = local_xg_curr;
    local_ygs[threadIdx.x] = local_yg_curr;

    // Wait for local arrays to be filled
    __syncthreads();

    // Compute global values off of local ones, but only in thread 0
    if (threadIdx.x != 0) {
        return;
    }
    // Initialize current global values
    double global_wmin_curr = local_wmins[0];
    double global_wmax_curr = local_wmaxs[0];
    double global_wsum_curr = local_wsums[0];
    double global_xg_curr = local_xgs[0];
    double global_yg_curr = local_ygs[0];
    // Loop to compute global values
    for (i = 1; i < blockDim.x; i++) {
        // Update sums
        global_wsum_curr += local_wsums[i];
        global_xg_curr += local_xgs[i];
        global_yg_curr += local_ygs[i];
        // Update optima
        if (global_wmin_curr > local_wmins[i])
            global_wmin_curr = local_wmins[i];
        if (global_wmax_curr < local_wmaxs[i])
            global_wmax_curr = local_wmaxs[i];
    }

    // Assign values to return array
    returns[0] = global_wmin_curr;
    returns[1] = global_wmax_curr;
    returns[2] = global_wsum_curr;
    returns[3] = global_xg_curr / global_wsum_curr;
    returns[4] = global_yg_curr / global_wsum_curr;
    return;
}



void ParticleStats(struct Population * p, int t) {
    /*
     * write a file with statistics on population
    */

    FILE *stats;
    double returns[5];
    double *stats_dev;

    cudaMalloc(&stats_dev, 5 * sizeof(double));

    double *weight_temp, *x_temp, *y_temp, *vx_temp, *vy_temp;
    Population *p_dev;


    cudaMalloc(&p_dev, sizeof(struct Population));
    //weight
    cudaMalloc(&weight_temp, p->np * sizeof(double));
    cudaMemcpy(&(p_dev->weight), &weight_temp, sizeof(double *), cudaMemcpyHostToDevice);

    weight_temp = NULL;
    cudaMalloc(&weight_temp, p->np * sizeof(double));
    cudaMemcpy(weight_temp, p->weight, p->np * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(p_dev->weight), &weight_temp, sizeof(double *), cudaMemcpyHostToDevice);

    //x
    cudaMalloc(&x_temp, p->np * sizeof(double));
    cudaMemcpy(&(p_dev->x), &x_temp, sizeof(double *), cudaMemcpyHostToDevice);

    x_temp = NULL;
    cudaMalloc(&x_temp, p->np * sizeof(double));
    cudaMemcpy(x_temp, p->x, p->np * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(p_dev->x), &x_temp, sizeof(double *), cudaMemcpyHostToDevice);

    //y
    cudaMalloc(&y_temp, p->np * sizeof(double));
    cudaMemcpy(&(p_dev->y), &y_temp, sizeof(double *), cudaMemcpyHostToDevice);

    y_temp = NULL;
    cudaMalloc(&y_temp, p->np * sizeof(double));
    cudaMemcpy(y_temp, p->y, p->np * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(p_dev->y), &y_temp, sizeof(double *), cudaMemcpyHostToDevice);

    //vx
    cudaMalloc(&vx_temp, p->np * sizeof(double));
    cudaMemcpy(&(p_dev->vx), &vx_temp, sizeof(double *), cudaMemcpyHostToDevice);

    vx_temp = NULL;
    cudaMalloc(&vx_temp, p->np * sizeof(double));
    cudaMemcpy(vx_temp, p->vx, p->np * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(p_dev->vx), &vx_temp, sizeof(double *), cudaMemcpyHostToDevice);

    //vy
    cudaMalloc(&vy_temp, p->np * sizeof(double));
    cudaMemcpy(&(p_dev->vy), &vy_temp, sizeof(double *), cudaMemcpyHostToDevice);

    vy_temp = NULL;
    cudaMalloc(&vy_temp, p->np * sizeof(double));
    cudaMemcpy(vy_temp, p->vy, p->np * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(p_dev->vy), &vy_temp, sizeof(double *), cudaMemcpyHostToDevice);

    if (t <= 0) stats = fopen("Population.sta", "w");
    else stats = fopen("Population.sta", "a"); // append new data
    if (stats == NULL) {
        fprintf(stderr, "Error append/open file Population.sta\n");
        exit(1);
    }
    int n_threads = sqrt(p->np);  // minimum for x + N/x
    if (n_threads > SHARED_MEM_MAX_THREADS)
        n_threads = SHARED_MEM_MAX_THREADS;
    ParallelComputeStats<<<1, n_threads>>>(p_dev, stats_dev);
    cudaDeviceSynchronize();

    cudaMemcpy(returns, stats_dev, 5 * sizeof(double), cudaMemcpyDeviceToHost);
    // note: stats = [wmin, wmax, w, xg, yg]

    fprintf(stats, "At iteration %d particles: %d; wmin, wmax = %lf, %lf;\n",
            t, p->np, returns[0], returns[1]);
    fprintf(stats, "   total weight = %lf; CM = (%10.4lf,%10.4lf)\n",
            returns[2], returns[3], returns[4]);
    fclose(stats);

    cudaFree(stats_dev);
    cudaFree(p_dev);
    cudaFree(weight_temp);
    cudaFree(x_temp);
    cudaFree(y_temp);
    cudaFree(vx_temp);
    cudaFree(vy_temp);
}



// Functions prototypes
int rowlen(char *riga);

int readrow(char *rg, int nc, FILE *daleg);

void ParticleScreen(struct i2dGrid *pgrid, struct Population * pp, int step, double rmin, double rmax);

__global__ void MinMaxIntVal(int total_size, int *values, int *min, int *max);

__global__ void MinMaxDoubleVal(int total_size, double *values, double *min, double *max);

void IntVal2ppm(int s1, int s2, int *idata, int *vmin, int *vmax, char *name);



__global__ void ComptPopulation(struct Population *p, double *forces_0, double *forces_1, double timebit) {
    /*
     * compute effects of forces on particles in a interval time
     *
    */
    int i;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stridex = gridDim.x * blockDim.x;

    if (idx >= p->np){
        return;
    }

    for (i = idx; i < p->np; i+=stridex) {
        p->x[i] = p->x[i] + (p->vx[i] * timebit) +
                  (0.5 * forces_0[i] * timebit * timebit / p->weight[i]);
        p->vx[i] = p->vx[i] + forces_1[i] * timebit / p->weight[i];
        p->y[i] = p->y[i] + (p->vy[i] * timebit) +
                  (0.5 * forces_0[i] * timebit * timebit / p->weight[i]);
        p->vy[i] = p->vy[i] + forces_1[i] * timebit / p->weight[i];
    }
}



void InitGrid(char *InputFile) {

    int nv, iv;
    double dv;
    char filerow[80];
    FILE *inpunit;

    fprintf(stdout, "Initializing grids ...\n");

    inpunit = fopen(InputFile, "r");
    if (!inpunit) {
        fprintf(stderr, "!!!! Error read access to file %s\n", InputFile);
        exit(-1);
    }

    // Now read measured values; they are read in the following order:
    // GenFieldGrid.EX, GenFieldGrid.EY,
    // GenFieldGrid.Xs, GenFieldGrid.Xe, GenFieldGrid.Ys, GenFieldGrid.Ye
    // ParticleGrid.Xs, ParticleGrid.Xe, ParticleGrid.Ys, ParticleGrid.Ye

    nv = 0;
    iv = 0;
    dv = 0.0;
    while (1) {
        if (readrow(filerow, 80, inpunit) < 1) {
            fprintf(stderr, "Error reading input file\n");
            exit(-1);
        }
        if (filerow[0] == '#') continue;
        if (nv <= 0) {
            if (sscanf(filerow, "%d", &iv) < 1) {
                fprintf(stderr, "Error reading EX from string\n");
                exit(-1);
            }
            GenFieldGrid.EX = iv;
            nv = 1;
            continue;
        }
        if (nv == 1) {
            if (sscanf(filerow, "%d", &iv) < 1) {
                fprintf(stderr, "Error reading EY from string\n");
                exit(-1);
            }
            GenFieldGrid.EY = iv;
            nv++;
            continue;

        }
        if (nv == 2) {
            if (sscanf(filerow, "%lf", &dv) < 1) {
                fprintf(stderr, "Error reading GenFieldGrid.Xs from string\n");
                exit(-1);
            }
            GenFieldGrid.Xs = dv;
            nv++;
            continue;

        }
        if (nv == 3) {
            if (sscanf(filerow, "%lf", &dv) < 1) {
                fprintf(stderr, "Error reading GenFieldGrid.Xe from string\n");
                exit(-1);
            }
            GenFieldGrid.Xe = dv;
            nv++;
            continue;

        }
        if (nv == 4) {
            if (sscanf(filerow, "%lf", &dv) < 1) {
                fprintf(stderr, "Error reading GenFieldGrid.Ys from string\n");
                exit(-1);
            }
            GenFieldGrid.Ys = dv;
            nv++;
            continue;

        }
        if (nv == 5) {
            if (sscanf(filerow, "%lf", &dv) < 1) {
                fprintf(stderr, "Error reading GenFieldGrid.Ye from string\n");
                exit(-1);
            }
            GenFieldGrid.Ye = dv;
            nv++;
            continue;

        }
        if (nv <= 6) {
            if (sscanf(filerow, "%d", &iv) < 1) {
                fprintf(stderr, "Error reading ParticleGrid.EX from string\n");
                exit(-1);
            }
            ParticleGrid.EX = iv;
            nv++;
            continue;
        }
        if (nv == 7) {
            if (sscanf(filerow, "%d", &iv) < 1) {
                fprintf(stderr, "Error reading ParticleGrid.EY from string\n");
                exit(-1);
            }
            ParticleGrid.EY = iv;
            nv++;
            continue;

        }
        if (nv == 8) {
            if (sscanf(filerow, "%lf", &dv) < 1) {
                fprintf(stderr, "Error reading ParticleGrid.Xs from string\n");
                exit(-1);
            }
            ParticleGrid.Xs = dv;
            nv++;
            continue;

        }
        if (nv == 9) {
            if (sscanf(filerow, "%lf", &dv) < 1) {
                fprintf(stderr, "Error reading ParticleGrid.Xe from string\n");
                exit(-1);
            }
            ParticleGrid.Xe = dv;
            nv++;
            continue;

        }
        if (nv == 10) {
            if (sscanf(filerow, "%lf", &dv) < 1) {
                fprintf(stderr, "Error reading ParticleGrid.Ys from string\n");
                exit(-1);
            }
            ParticleGrid.Ys = dv;
            nv++;
            continue;

        }
        if (nv == 11) {
            if (sscanf(filerow, "%lf", &dv) < 1) {
                fprintf(stderr, "Error reading ParticleGrid.Ye from string\n");
                exit(-1);
            }
            ParticleGrid.Ye = dv;
            break;
        }
    }

    /*
      Now read MaxIters
    */
    MaxIters = 0;
    while (1) {
        if (readrow(filerow, 80, inpunit) < 1) {
            fprintf(stderr, "Error reading MaxIters from input file\n");
            exit(-1);
        }
        if (filerow[0] == '#' || rowlen(filerow) < 1) continue;
        if (sscanf(filerow, "%d", &MaxIters) < 1) {
            fprintf(stderr, "Error reading MaxIters from string\n");
            exit(-1);
        }
        printf("MaxIters = %d\n", MaxIters);
        break;
    }

    /*
      Now read MaxSteps
    */
    MaxSteps = 0;
    while (1) {
        if (readrow(filerow, 80, inpunit) < 1) {
            fprintf(stderr, "Error reading MaxSteps from input file\n");
            exit(-1);
        }
        if (filerow[0] == '#' || rowlen(filerow) < 1) continue;
        if (sscanf(filerow, "%d", &MaxSteps) < 1) {
            fprintf(stderr, "Error reading MaxSteps from string\n");
            exit(-1);
        }
        printf("MaxSteps = %d\n", MaxSteps);
        break;
    }

    /*
    ! Now read TimeBit
    */
    TimeBit = 0;
    while (1) {
        if (readrow(filerow, 80, inpunit) < 1) {
            fprintf(stderr, "Error reading TimeBit from input file\n");
            exit(-1);
        }
        if (filerow[0] == '#' || rowlen(filerow) < 1) continue;
        if (sscanf(filerow, "%lf", &TimeBit) < 1) {
            fprintf(stderr, "Error reading TimeBit from string\n");
            exit(-1);
        }
        printf("TimeBit = %lf\n", TimeBit);

        break;
    }

    fclose(inpunit);

    // Grid allocations
    iv = GenFieldGrid.EX * GenFieldGrid.EY;
    GenFieldGrid.Values = (int *)malloc(iv * sizeof(1));
    if (GenFieldGrid.Values == NULL) {
        fprintf(stderr, "Error allocating GenFieldGrid.Values \n");
        exit(-1);
    }
    iv = ParticleGrid.EX * ParticleGrid.EY;
    ParticleGrid.Values = (int *)malloc(iv * sizeof(1));
    if (ParticleGrid.Values == NULL) {
        fprintf(stderr, "Error allocating ParticleGrid.Values \n");
        exit(-1);
    }
    fprintf(stdout, "GenFieldGrid ");
    print_i2dGrid(GenFieldGrid);
    fprintf(stdout, "ParticleGrid ");
    print_i2dGrid(ParticleGrid);

    return;
}



__global__ void GeneratingField(struct i2dGrid *grid, int MaxIt, int *values) {
    /*
   !  Compute "generating" points
   !  Output:
   !    *grid.Values
   */

    int ix, iy, iz;
    double ca, cb, za, zb;
    double rad, zan, zbn;

    int Xdots = grid->EX;
    int Ydots = grid->EY;
    double Sr = grid->Xe - grid->Xs;
    double Si = grid->Ye - grid->Ys;
    double Ir = grid->Xs;
    double Ii = grid->Ys;
    double Xinc = Sr / (double) Xdots;
    double Yinc = Si / (double) Ydots;

    int izmn = 9999;
    int izmx = -9;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stridex = gridDim.x * blockDim.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int stridey = gridDim.y * blockDim.y;

    if (idx >= Xdots || idy >= Ydots){
        return;
    }

    for (iy = idy; iy < Ydots; iy+=stridey) {
        for (ix = idx; ix < Xdots; ix+=stridex) {
            ca = Xinc * ix + Ir;
            cb = Yinc * iy + Ii;
            rad = sqrt(ca * ca * ((double) 1.0 + (cb / ca) * (cb / ca)));
            zan = 0.0;
            zbn = 0.0;
            for (iz = 1; iz <= MaxIt; iz++) {
                if (rad > (double) 2.0) break;
                za = zan;
                zb = zbn;
                zan = ca + (za - zb) * (za + zb);
                zbn = 2.0 * (za * zb + cb / 2.0);
                rad = sqrt(zan * zan * ((double) 1.0 + (zbn / zan) * (zbn / zan)));
            }
            if (izmn > iz) izmn = iz;
            if (izmx < iz) izmx = iz;
            if (iz >= MaxIt) iz = 0;
            values[index2D(ix, iy, Xdots)] = iz;
        }
    }
    return;
}



void ParticleGeneration(struct i2dGrid *grid, struct i2dGrid *pgrid, struct Population *pp)
{
    // A system of particles is generated according to the value distribution of
    // grid.Values
    int v, ix, iy, np, n;
    double p;

    int Xdots = grid->EX;
    int Ydots = grid->EY;
    int N = Xdots * Ydots;

    // Initialize values_dev to host
    int *values_dev;
    cudaMalloc(&values_dev, N * sizeof(int));
    cudaMemcpy(values_dev, grid->Values, N * sizeof(int), cudaMemcpyHostToDevice);

    // Compute min and max values
    int vmin, vmax;
    int *vmin_dev, *vmax_dev;
    cudaMalloc(&vmin_dev, sizeof(int));
    cudaMalloc(&vmax_dev, sizeof(int));
    int n_threads = sqrt(N);  // minimum for x + N/x
    if (n_threads > SHARED_MEM_MAX_THREADS)
        n_threads = SHARED_MEM_MAX_THREADS;
    MinMaxIntVal<<<1, n_threads>>>(N, values_dev, vmin_dev, vmax_dev);  // shared memory only works in the same block
    cudaDeviceSynchronize();
    cudaMemcpy(&vmin, vmin_dev, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&vmax, vmax_dev, sizeof(int), cudaMemcpyDeviceToHost);

    // Just count number of particles to be generated
    vmin = (double) (1 * vmax + 29 * vmin) / 30.0;
    np = 0;
    for ( ix = 0; ix < Xdots; ix++ ) {
        for ( iy = 0; iy < Ydots; iy++ ) {
            v = grid->Values[index2D(ix,iy,Xdots)];
            if (vmin <= v && v <= vmax) {
                np++;
            }
        }
    }

    // allocate memory space for particles
    pp->np = np;
    pp->weight = (double *) malloc(np * sizeof(double));
    pp->x = (double *) malloc(np * sizeof(double));
    pp->y = (double *) malloc(np * sizeof(double));
    pp->vx = (double *) malloc(np * sizeof(double));
    pp->vy = (double *) malloc(np * sizeof(double));

    // Population initialization
    n = 0;
    for ( ix = 0; ix < grid->EX; ix++ ) {
        for ( iy = 0; iy < grid->EY; iy++ ) {
            v = grid->Values[index2D(ix,iy,Xdots)];
            if ( v <= vmax && v >= vmin ) {
                pp->weight[n] = v*10.0;

                p = (pgrid->Xe-pgrid->Xs) * ix / (grid->EX * 2.0);
                pp->x[n] = pgrid->Xs + ((pgrid->Xe-pgrid->Xs)/4.0) + p;

                p = (pgrid->Ye-pgrid->Ys) * iy / (grid->EY * 2.0);
                pp->y[n] = pgrid->Ys + ((pgrid->Ye-pgrid->Ys)/4.0) + p;

                pp->vx[n] = pp->vy[n] = 0.0; // at start particles are still

                n++;
                if ( n >= np )
                    break;
            }
        }
        if ( n >= np )
            break;
    }

    print_Population(*pp);
    cudaFree(values_dev);
    cudaFree(vmin_dev);
    cudaFree(vmax_dev);
} // end ParticleGeneration



__global__ void SystemInstantEvolution(struct Population *pp, double *forces_0, double *forces_1) {

    int i, j;
    //variables to compute force between p1 and p2
    double force, d2, dx, dy, f0_tot, f1_tot;
    static double k = 0.001, tiny = (double) 1.0 / (double) 1000000.0;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stridex = gridDim.x * blockDim.x;

    for (i = idx; i < pp->np; i+=stridex) {
        f0_tot = 0.0;
        f1_tot = 0.0;
        for (j = 0; j < pp->np; j++) {
            if (j != i) {
                // compute force between p1 and p2
                dx = pp->x[j] - pp->x[i];
                dy = pp->y[j] - pp->y[i];
                d2 = dx * dx + dy * dy;  // what if particles get in touch? Simply avoid the case
                if (d2 < tiny) d2 = tiny;
                force = (k * pp->weight[i] * pp->weight[j]) / d2;
                f0_tot += force * dx / sqrt(d2);
                f1_tot += force * dy / sqrt(d2);
            }
        }
        forces_0[i] = f0_tot;
        forces_1[i] = f1_tot;
    }
    return;
}



void SystemEvolution(struct i2dGrid *pgrid, struct Population *pp, int mxiter, double timebit) {
    int t;

    double *g_forces_0;
    double *g_forces_1;
    cudaMalloc(&g_forces_0, pp->np * sizeof(double));
    cudaMalloc(&g_forces_1, pp->np * sizeof(double));

    Population *pp_dev;
    double *temp_dev, *weight_temp, *x_temp, *y_temp, *vx_temp, *vy_temp;

    cudaMalloc(&pp_dev, sizeof(struct Population));
    cudaMemcpy(&(pp_dev->np), &(pp->np), sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&weight_temp, pp->np * sizeof(double));
    cudaMemcpy(&(pp_dev->weight), &weight_temp, sizeof(double *), cudaMemcpyHostToDevice);

    weight_temp = NULL;
    cudaMalloc(&weight_temp, pp->np * sizeof(double));
    cudaMemcpy(weight_temp, pp->weight, pp->np * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(pp_dev->weight), &weight_temp, sizeof(double *), cudaMemcpyHostToDevice);

    dim3 threads_per_block_uni (1024, 1, 1); // maximum number of threads per block
    dim3 number_of_blocks_uni (32 * num_SMs, 1, 1);

    // Initialize weights_dev to host
    double *weights_dev;
    cudaMalloc(&weights_dev, pp->np * sizeof(double));
    cudaMemcpy(weights_dev, pp->weight, pp->np * sizeof(double), cudaMemcpyHostToDevice);

    // Compute min and max values to pass to ParticleScreen
    double rmin, rmax;
    double *rmin_dev, *rmax_dev;
    cudaMalloc(&rmin_dev, sizeof(double));
    cudaMalloc(&rmax_dev, sizeof(double));
    int n_threads = sqrt(pp->np);  // minimum for x + N/x
    if (n_threads > SHARED_MEM_MAX_THREADS)
        n_threads = SHARED_MEM_MAX_THREADS;
    MinMaxDoubleVal<<<1, n_threads>>>(pp->np, weights_dev, rmin_dev, rmax_dev);  // shared memory only works in the same block
    cudaDeviceSynchronize();
    cudaMemcpy(&rmin, rmin_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&rmax, rmax_dev, sizeof(double), cudaMemcpyDeviceToHost);

    // compute forces acting on each particle step by step
    for (t = 0; t < mxiter; t++) {
        fprintf(stdout, "Step %d of %d\n", t, mxiter);
        ParticleScreen(pgrid, pp, t, rmin, rmax);
        // DumpPopulation call frequency may be changed
        if (t % 4 == 0) DumpPopulation(*pp, t);
        ParticleStats(pp, t);

        //x
        cudaMalloc(&x_temp, pp->np * sizeof(double));
        cudaMemcpy(&(pp_dev->x), &x_temp, sizeof(double *), cudaMemcpyHostToDevice);

        x_temp = NULL;
        cudaMalloc(&x_temp, pp->np * sizeof(double));
        cudaMemcpy(x_temp, pp->x, pp->np * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(&(pp_dev->x), &x_temp, sizeof(double *), cudaMemcpyHostToDevice);

        //y
        cudaMalloc(&y_temp, pp->np * sizeof(double));
        cudaMemcpy(&(pp_dev->y), &y_temp, sizeof(double *), cudaMemcpyHostToDevice);

        y_temp = NULL;
        cudaMalloc(&y_temp, pp->np * sizeof(double));
        cudaMemcpy(y_temp, pp->y, pp->np * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(&(pp_dev->y), &y_temp, sizeof(double *), cudaMemcpyHostToDevice);

        //vx
        cudaMalloc(&vx_temp, pp->np * sizeof(double));
        cudaMemcpy(&(pp_dev->vx), &vx_temp, sizeof(double *), cudaMemcpyHostToDevice);

        vx_temp = NULL;
        cudaMalloc(&vx_temp, pp->np * sizeof(double));
        cudaMemcpy(vx_temp, pp->vx, pp->np * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(&(pp_dev->vx), &vx_temp, sizeof(double *), cudaMemcpyHostToDevice);

        //vy
        cudaMalloc(&vy_temp, pp->np * sizeof(double));
        cudaMemcpy(&(pp_dev->vy), &vy_temp, sizeof(double *), cudaMemcpyHostToDevice);

        vy_temp = NULL;
        cudaMalloc(&vy_temp, pp->np * sizeof(double));
        cudaMemcpy(vy_temp, pp->vy, pp->np * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(&(pp_dev->vy), &vy_temp, sizeof(double *), cudaMemcpyHostToDevice);

        SystemInstantEvolution<<<number_of_blocks_uni, threads_per_block_uni>>>(pp_dev, g_forces_0, g_forces_1);
        cudaDeviceSynchronize();

        ComptPopulation<<<number_of_blocks_uni, threads_per_block_uni>>>(pp_dev, g_forces_0, g_forces_1, timebit);
        cudaDeviceSynchronize();


        cudaMalloc(&temp_dev, pp->np * sizeof(double));
        cudaMemcpy(&temp_dev, &(pp_dev->weight), sizeof(double *), cudaMemcpyDeviceToHost);
        cudaMemcpy(pp->weight, temp_dev, pp->np * sizeof(double), cudaMemcpyDeviceToHost);

        cudaMalloc(&temp_dev, pp->np * sizeof(double));
        cudaMemcpy(&temp_dev, &(pp_dev->x), sizeof(double *), cudaMemcpyDeviceToHost);
        cudaMemcpy(pp->x, temp_dev, pp->np * sizeof(double), cudaMemcpyDeviceToHost);

        cudaMalloc(&temp_dev, pp->np * sizeof(double));
        cudaMemcpy(&temp_dev, &(pp_dev->y), sizeof(double *), cudaMemcpyDeviceToHost);
        cudaMemcpy(pp->y, temp_dev, pp->np * sizeof(double), cudaMemcpyDeviceToHost);

        cudaMalloc(&temp_dev, pp->np * sizeof(double));
        cudaMemcpy(&temp_dev, &(pp_dev->vx), sizeof(double *), cudaMemcpyDeviceToHost);
        cudaMemcpy(pp->vx, temp_dev, pp->np * sizeof(double), cudaMemcpyDeviceToHost);

        cudaMalloc(&temp_dev, pp->np * sizeof(double));
        cudaMemcpy(&temp_dev, &(pp_dev->vy), sizeof(double *), cudaMemcpyDeviceToHost);
        cudaMemcpy(pp->vy, temp_dev, pp->np * sizeof(double), cudaMemcpyDeviceToHost);
    }

    cudaFree(g_forces_0);
    cudaFree(g_forces_1);
    cudaFree(pp_dev);
    cudaFree(weight_temp);
    cudaFree(weights_dev);
    cudaFree(rmin_dev);
    cudaFree(rmax_dev);
    cudaFree(x_temp);
    cudaFree(y_temp);
    cudaFree(vx_temp);
    cudaFree(vy_temp);
    cudaFree(temp_dev);  // TODO why does this raise an error?
}   // end SystemEvolution



__global__ void InitializeEmptyGridInt(int EX, int EY, int * values){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stridex = gridDim.x * blockDim.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int stridey = gridDim.y * blockDim.y;

    if (idx >= EX || idy >= EY) {
        return;
    }

    for (int iy = idy; iy < EY; iy+=stridey) {
        for (int ix = idx; ix < EX; ix+=stridex) {
            values[index2D(ix, iy, EX)] = 0;
        }
    }
}



void ParticleScreen(struct i2dGrid *pgrid, struct Population * pp, int step, double rmin, double rmax) {
    // Distribute a particle population in a grid for visualization purposes
    int ix, iy, n, wp;
    int static vmin, vmax;
    double Dx, Dy, wint, wv;
    char name[40];

    int Xdots = pgrid->EX;
    int Ydots = pgrid->EY;
    int N = pgrid->EX * pgrid->EY;

    dim3 threads_per_block (32, 32, 1);
    dim3 number_of_blocks (2, 2, 1); // (2 * 80) < 65535, maximum number of blocks per grid dimension

    // initialization of particlegrid.values in device
    int * v_host;
    v_host = (int *) malloc(N * sizeof(int));
    int * v_dev;
    cudaMalloc(&v_dev, N * sizeof(int));
    cudaMemcpy(v_dev, pgrid->Values, N * sizeof(int), cudaMemcpyHostToDevice);

    // TODO?: input should be pointer to i2grid and not to int[], either change the signature or add memcpy to new i2grid * pointer
    InitializeEmptyGridInt<<<number_of_blocks, threads_per_block>>>(pgrid->EX, pgrid->EY, v_dev);
    cudaDeviceSynchronize();

    cudaMemcpy(v_host, v_dev, N * sizeof(int), cudaMemcpyDeviceToHost);
    pgrid->Values = v_host;

    wint = rmax - rmin;
    Dx = pgrid->Xe - pgrid->Xs;
    Dy = pgrid->Ye - pgrid->Ys;

    for (n = 0; n < pp->np; n++) {
        // keep a tiny border free anyway
        ix = Xdots * pp->x[n] / Dx;
        if (ix >= Xdots - 1 || ix <= 0) continue;
        iy = Ydots * pp->y[n] / Dy;
        if (iy >= Ydots - 1 || iy <= 0) continue;
        wv = pp->weight[n] - rmin;
        wp = 10.0 * wv / wint;
        pgrid->Values[index2D(ix, iy, Xdots)] = wp;
        pgrid->Values[index2D(ix - 1, iy, Xdots)] = wp;
        pgrid->Values[index2D(ix + 1, iy, Xdots)] = wp;
        pgrid->Values[index2D(ix, iy - 1, Xdots)] = wp;
        pgrid->Values[index2D(ix, iy + 1, Xdots)] = wp;
    }
    sprintf(name, "stage%3.3d\0", step);
    if (step <= 0) { vmin = vmax = 0; }
    IntVal2ppm(pgrid->EX, pgrid->EY, pgrid->Values, &vmin, &vmax, name);

    cudaFree(v_dev);
} // end ParticleScreen



__global__ void MinMaxIntVal(int total_size, int *values, int *min, int *max)
{
    if (threadIdx.x >= blockDim.x) {
        return;
    }

    // Declare shared memory arrays for local optima
    __shared__ int local_mins[SHARED_MEM_MAX_THREADS];
    __shared__ int local_maxs[SHARED_MEM_MAX_THREADS];

    // Initialize size of data chunk for this thread, while adjusting for case of
    // non-exact division
    int local_size = total_size / blockDim.x;
    int remainder = total_size % blockDim.x;
    if (remainder != 0 && threadIdx.x < remainder)
        local_size++;

    // Initialize start and end data indexes for this thread, while adjusting for
    // case of non-exact division
    int first_val_idx = threadIdx.x * local_size;
    if (threadIdx.x >= remainder)
        first_val_idx += remainder;
    int last_val_idx = first_val_idx + local_size;

    // Initialize current local optima
    int min_loc_curr = values[first_val_idx];
    int max_loc_curr = values[first_val_idx];

    // Compute each of the blockDim.x local optima
    int i;
    for (i = first_val_idx+1; i < last_val_idx; i++) {
        if (min_loc_curr > values[i])
            min_loc_curr = values[i];
        if (max_loc_curr < values[i])
            max_loc_curr = values[i];
    }
    local_mins[threadIdx.x] = min_loc_curr;
    local_maxs[threadIdx.x] = max_loc_curr;

    // Wait for local optima arrays to be filled
    __syncthreads();

  // Find global optima among local ones, but only in thread 0
    if (threadIdx.x == 0) {
        int min_glob_curr = local_mins[0];
        int max_glob_curr = local_maxs[0];
        for (i = 1; i < blockDim.x; i++) {
            if (min_glob_curr > local_mins[i])
                min_glob_curr = local_mins[i];
            if (max_glob_curr < local_maxs[i])
                max_glob_curr = local_maxs[i];
        }
        *min = min_glob_curr;
        *max = max_glob_curr;
    }
    return;
}



__global__ void MinMaxDoubleVal(int total_size, double *values, double *min, double *max)
{

    if (threadIdx.x >= blockDim.x) {
        return;
    }

    // Declare shared memory arrays for local optima
    __shared__ double local_mins[SHARED_MEM_MAX_THREADS];
    __shared__ double local_maxs[SHARED_MEM_MAX_THREADS];

    // Initialize size of data chunk for this thread, while adjusting for case of
    // non-exact division
    int local_size = total_size / blockDim.x;
    int remainder = total_size % blockDim.x;
    if (remainder != 0 && threadIdx.x < remainder)
        local_size++;

    // Initialize start and end data indexes for this thread, while adjusting for
    // case of non-exact division
    int first_val_idx = threadIdx.x * local_size;
    if (threadIdx.x >= remainder)
        first_val_idx += remainder;
    int last_val_idx = first_val_idx + local_size;

    // Initialize current local optima
    double min_loc_curr = values[first_val_idx];
    double max_loc_curr = values[first_val_idx];

    // Compute each of the blockDim.x local optima
    int i;
    for (i = first_val_idx+1; i < last_val_idx; i++) {
        if (min_loc_curr > values[i])
            min_loc_curr = values[i];
        if (max_loc_curr < values[i])
           max_loc_curr = values[i];
    }
    local_mins[threadIdx.x] = min_loc_curr;
    local_maxs[threadIdx.x] = max_loc_curr;

    // Wait for local optima arrays to be filled
    __syncthreads();

    // Find global optima among local ones, but only in thread 0
    if (threadIdx.x == 0) {
        double min_glob_curr = local_mins[0];
        double max_glob_curr = local_maxs[0];
        for (i = 1; i < blockDim.x; i++) {
            if (min_glob_curr > local_mins[i])
                min_glob_curr = local_mins[i];
            if (max_glob_curr < local_maxs[i])
                  max_glob_curr = local_maxs[i];
        }
        *min = min_glob_curr;
        *max = max_glob_curr;
    }

    return;
}



int rowlen(char *riga) {
    int lungh;
    char c;

    lungh = strlen(riga);
    while (lungh > 0) {
        lungh--;
        c = *(riga + lungh);
        if (c == '\0') continue;
        if (c == '\40') continue;     /*  space  */
        if (c == '\b') continue;
        if (c == '\f') continue;
        if (c == '\r') continue;
        if (c == '\v') continue;
        if (c == '\n') continue;
        if (c == '\t') continue;
        return (lungh + 1);
    }
    return (0);
}



int readrow(char *rg, int nc, FILE *daleg) {
    //int rowlen(), lrg;
    int lrg;

    if (fgets(rg, nc, daleg) == NULL) return (-1);
    lrg = rowlen(rg);
    if (lrg < nc) {
        rg[lrg] = '\0';
        lrg++;
    }
    return (lrg);
}



void IntVal2ppm(int s1, int s2, int *idata, int *vmin, int *vmax, char *name) {
    /*
       Simple subroutine to dump double data with fixed min & max values
          in a PPM format
    */
    int i, j;
    int cm[3][256];  /* R,G,B, Colour Map */
    FILE *ouni, *ColMap;
    int vp, vs;
    int rmin, rmax, value;
    char fname[80];
    // char jname[80], command[80];
    /*
       Define color map: 256 colours
    */
    ColMap = fopen("ColorMap.txt", "r");
    if (ColMap == NULL) {
        fprintf(stderr, "Error read opening file ColorMap.txt\n");
        exit(-1);
    }
    for (i = 0; i < 256; i++) {
        if (fscanf(ColMap, " %3d %3d %3d",
                   &cm[0][i], &cm[1][i], &cm[2][i]) < 3) {
            fprintf(stderr, "Error reading colour map at line %d: r, g, b =", (i + 1));
            fprintf(stderr, " %3.3d %3.3d %3.3d\n", cm[0][i], cm[1][i], cm[2][i]);
            exit(1);
        }
    }
    /*
       Write on unit 700 with  PPM format
    */
    strcpy(fname, name);
    strcat(fname, ".ppm\0");
    ouni = fopen(fname, "w");
    if (!ouni) {
        fprintf(stderr, "!!!! Error write access to file %s\n", fname);
    }
    /*  Magic code */
    fprintf(ouni, "P3\n");
    /*  Dimensions */
    fprintf(ouni, "%d %d\n", s1, s2);
    /*  Maximum value */
    fprintf(ouni, "255\n");
    /*  Values from 0 to 255 */

    int *rmin_dev, *rmax_dev, *v_dev;
    int N = s1 * s2;
    cudaMalloc(&rmin_dev, sizeof(int));
    cudaMalloc(&rmax_dev, sizeof(int));
    cudaMalloc(&v_dev, N * sizeof(int));
    cudaMemcpy(v_dev, idata, N * sizeof(int), cudaMemcpyHostToDevice);

    // Run kernel with optimal number of threads
    int n_threads = sqrt(N);  // minimum for x + N/x
    if (n_threads > SHARED_MEM_MAX_THREADS)
        n_threads = SHARED_MEM_MAX_THREADS;

    MinMaxIntVal<<<1, n_threads>>>(N, v_dev, rmin_dev, rmax_dev);  // shared memory only works in the same block
    cudaDeviceSynchronize();

    cudaMemcpy(&rmin, rmin_dev, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&rmax, rmax_dev, sizeof(int), cudaMemcpyDeviceToHost);

    if ((*vmin == *vmax) && (*vmin == 0)) {
        *vmin = rmin;
        *vmax = rmax;
    } else {
        rmin = *vmin;
        rmax = *vmax;
    }
    vs = 0;
    for (i = 0; i < s1; i++) {
        for (j = 0; j < s2; j++) {
            value = idata[i * s2 + j];
            if (value < rmin) value = rmin;
            if (value > rmax) value = rmax;
            vp = (int) ((double) (value - rmin) * (double) 255.0 / (double) (rmax - rmin));
            vs++;
            fprintf(ouni, " %3.3d %3.3d %3.3d", cm[0][vp], cm[1][vp], cm[2][vp]);
            if (vs >= 10) {
                fprintf(ouni, " \n");
                vs = 0;
            }
        }
        fprintf(ouni, " ");
        vs = 0;
    }
    fclose(ouni);
    // the following instructions require ImageMagick tool: comment out if not available
    // strcpy(jname, name);
    // strcat(jname, ".jpg\0");
    // sprintf(command, "convert %s %s\0", fname, jname);
    // system(command);

    cudaFree(rmin_dev);
    cudaFree(rmax_dev);
    cudaFree(v_dev);

    return;
} // end IntVal2ppm



int main(int argc, char *argv[]){
    // Define GPU dimensions with number of Streaming Multiprocessors, to further improve performance
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, deviceId);
    dim3 threads_per_block (32, 32, 1);  // 32 * 32 = 1024, maximum number of threads per block
    dim3 number_of_blocks (32 * num_SMs, 32 * num_SMs, 1);  // (32 * 80) < 65535, maximum number of blocks per grid dimension

    // Start clock
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    // Initialize grid
    InitGrid("Particles.inp");

    // Copy GenFieldGrid to device
    cudaMalloc(&GenFieldGrid_dev, sizeof(struct i2dGrid));
    cudaMemcpy(GenFieldGrid_dev, &GenFieldGrid, sizeof(struct i2dGrid), cudaMemcpyHostToDevice);
    // Initialize separate contained for GenFieldGrid values
    int N = GenFieldGrid.EX * GenFieldGrid.EY;
    int *values_dev;
    cudaMalloc(&values_dev, N * sizeof(int));

    // Generate field
    printf("GeneratingField...\n");
    GeneratingField <<<number_of_blocks, threads_per_block>>> (GenFieldGrid_dev, MaxIters, values_dev);
    cudaDeviceSynchronize();
    // Copy generated values back to GenFieldGrid
    cudaMemcpy(GenFieldGrid.Values, values_dev, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Initialize particles
    printf("ParticleGeneration...\n");
    ParticleGeneration(&GenFieldGrid, &ParticleGrid, &Particles);

    // Compute evolution of the particle population
    printf("SystemEvolution...\n");
    SystemEvolution (&ParticleGrid, &Particles, MaxSteps, TimeBit);

    // Free device memory
    cudaFree(GenFieldGrid_dev);
    cudaFree(values_dev);

    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    unsigned long long microseconds = (t1.tv_sec - t0.tv_sec) * 1000000 + (t1.tv_nsec - t0.tv_nsec) / 1000;
    printf("Computations ended in %lu microseconds\n", microseconds);

    fprintf(stdout, "End of program!\n");

    return (0);
}  // end FinalApplication
