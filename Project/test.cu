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

#define SHARED_MEM_MAX_THREADS 1024

struct i2dGrid {
    int EX, EY; // extensions in X and Y directions
    double Xs, Xe, Ys, Ye; // initial and final value for X and Y directions
    int *Values; // 2D matrix of values
} GenFieldGrid, ParticleGrid;

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
    int * pop_count;
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

    double * temp_dev;
    Population * p_dev;

    cudaMalloc(&temp_dev, p->np * sizeof(double));
    cudaMemcpy(temp_dev, p->weight, p->np * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(p_dev->weight), &temp_dev, sizeof(double *), cudaMemcpyHostToDevice);

    temp_dev = NULL;
    cudaMalloc(&temp_dev, p->np * sizeof(double));
    cudaMemcpy(temp_dev, p->x, p->np * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(p_dev->x), &temp_dev, sizeof(double *), cudaMemcpyHostToDevice);

    temp_dev = NULL;
    cudaMalloc(&temp_dev, p->np * sizeof(double));
    cudaMemcpy(temp_dev, p->y, p->np * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(p_dev->y), &temp_dev, sizeof(double *), cudaMemcpyHostToDevice);

    temp_dev = NULL;
    cudaMalloc(&temp_dev, p->np * sizeof(double));
    cudaMemcpy(temp_dev, p->vx, p->np * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(p_dev->vx), &temp_dev, sizeof(double *), cudaMemcpyHostToDevice);

    temp_dev = NULL;
    cudaMalloc(&temp_dev, p->np * sizeof(double));
    cudaMemcpy(temp_dev, p->vy, p->np * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(p_dev->vy), &temp_dev, sizeof(double *), cudaMemcpyHostToDevice);

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
    cudaMemcpy(stats_dev, returns, 5 * sizeof(double), cudaMemcpyDeviceToHost);
    // note: stats = [wmin, wmax, w, xg, yg]

    fprintf(stats, "At iteration %d particles: %d; wmin, wmax = %lf, %lf;\n",
            t, p->np, returns[0], returns[1]);
    fprintf(stats, "   total weight = %lf; CM = (%10.4lf,%10.4lf)\n",
            returns[2], returns[3], returns[4]);
    fclose(stats);

    cudaFree(stats_dev);
    cudaFree(temp_dev);
}


#define index2D(i, j, LD1) i + ((j)*LD1)    // element position in 2-D arrays

// Parameters
int MaxIters, MaxSteps;

double TimeBit;   // Evolution time steps
double * TimeBit_dev;

i2dGrid * GenFieldGrid_dev;
i2dGrid * ParticleGrid_dev;
//  functions  prototypes
int rowlen(char *riga);

int readrow(char *rg, int nc, FILE *daleg);

void InitGrid(char *InputFile);

void ParticleScreen(struct i2dGrid *pgrid, struct Population * pp, int s, double rmin, double rmax);

__global__ void MinMaxIntVal(int total_size, int *values, int *min, int *max);

__global__ void MinMaxDoubleVal(int total_size, double *values, double *min, double *max);

void IntVal2ppm(int s1, int s2, int *idata, int *vmin, int *vmax, char *name);

__global__ void newparticle(struct particle *p, double weight, double x, double y, double vx, double vy);

__global__ void GeneratingField(struct i2dGrid *grid, int * iterations, int * values);

__global__ void CountPopulation(int total_size, int *values, int *count, int vmin, int vmax);

__global__ void ParticleGeneration(struct i2dGrid * grid, struct i2dGrid * pgrid, struct Population *pp, int * values, int vmin, int vmax);

void SystemEvolution(struct i2dGrid *pgrid, struct Population *pp, int mxiter, double timebit, double min, double max);

__global__ void ForceCompt(double *f, struct particle p1, struct particle p2);


__global__ void newparticle(struct particle *p, double weight, double x, double y, double vx, double vy) {
    /*
     * define a new object with passed parameters
    */
    p->weight = weight;
    p->x = x;
    p->y = y;
    p->vx = vx;
    p->vy = vy;

}


__global__ void ForceCompt(double * f, struct particle p1, struct particle p2) {
    /*
     * Compute force acting on p1 by p1-p2 interactions
     *
    */
	double force, d2, dx, dy;
    static double k = 0.001, tiny = (double) 1.0 / (double) 1000000.0;

    dx = p2.x - p1.x;
    dy = p2.y - p1.y;
    d2 = dx * dx + dy * dy;  // what if particles get in touch? Simply avoid the case
    if (d2 < tiny) d2 = tiny;
    force = (k * p1.weight * p2.weight) / d2;
    f[0] = force * dx / sqrt(d2);
    f[1] = force * dy / sqrt(d2);
}

__global__ void ComptPopulation(struct Population *p, double *forces, double timebit) {
    /*
     * compute effects of forces on particles in a interval time
     *
    */
    int i;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stridex = gridDim.x * blockDim.x;
    for (i = idx; i < p->np; i+=stridex) {
	    p->x[i] = p->x[i] + (p->vx[i] * timebit) +
                  (0.5 * forces[index2D(0, i, 2)] * timebit * timebit / p->weight[i]);
        p->vx[i] = p->vx[i] + forces[index2D(0, i, 2)] * timebit / p->weight[i];
        p->y[i] = p->y[i] + (p->vy[i] * timebit) +
                  (0.5 * forces[index2D(1, i, 2)] * timebit * timebit / p->weight[i]);
        p->vy[i] = p->vy[i] + forces[index2D(1, i, 2)] * timebit / p->weight[i];
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


__global__ void GeneratingField(struct i2dGrid *grid, int MaxIt, int * values) {
    /*
   !  Compute "generating" points
   !  Output:
   !    *grid.Values
   */

    int ix, iy, iz;
    double ca, cb, za, zb;
    double rad, zan, zbn;
    double Xinc, Sr, Ir;
    int izmn, izmx;
    int Xdots, Ydots;

    Xdots = grid->EX;
    Ydots = grid->EY;
    Sr = grid->Xe - grid->Xs;
    Ir = grid->Xs;
    Xinc = Sr / (double) Xdots;

    izmn = 9999;
    izmx = -9;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stridex = gridDim.x * blockDim.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int stridey = gridDim.y * blockDim.y;

    for (iy = idy; iy < Ydots; iy+=stridey) {
        for (ix = idx; ix < Xdots; ix+=stridex) {
		ca = Xinc * ix + Ir;
            rad = sqrt(ca * ca * ((double) 1.0 + (cb / ca) * (cb / ca)));
            zan = 0.0;
            zbn = 0.0;
            //OPT: computation depends on previously calculated values, cannot be parallelized
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


__global__ void CountPopulation(int total_size, int *values, int *count, int vmin, int vmax)
{
    if (threadIdx.x >= blockDim.x) {
        return;
    }

    // Declare shared memory arrays for local counts
    __shared__ int local_counts[SHARED_MEM_MAX_THREADS];

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

    // Compute each of the blockDim.x local counts
    int i;
    for (i = first_val_idx; i < last_val_idx; i++) {
        if (vmin <= values[i] && values[i] <= vmax)
            local_counts[threadIdx.x]++;
    }

    // Wait for local optima arrays to be filled
    __syncthreads();

    // Compute global count, but only in thread 0
    if (threadIdx.x == 0) {
        *count = local_counts[0];
        for (i = 1; i < blockDim.x; i++) {
            *count += local_counts[i];
        }
    }

    return;
}


__global__ void ParticleGeneration(struct i2dGrid * grid, struct i2dGrid * pgrid, struct Population *pp, int * values, int vmin, int vmax) {
    // A system of particles is generated according to the value distribution of grid.Values
    int v;
    int Xdots, Ydots;
    int ix, iy, np, n;
    double p;

    Xdots = grid->EX;
    Ydots = grid->EY;

    // Just count number of particles to be generated
    vmin = (double) (1 * vmax + 29 * vmin) / 30.0;
    np = pp->np;
    n = 0;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stridex = gridDim.x * blockDim.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int stridey = gridDim.y * blockDim.y;

    for (iy = idy; iy < Ydots; iy+=stridey) {
        for (ix = idx; ix < Xdots; ix+=stridex) {
	    v = values[index2D(ix, iy, Xdots)];
	    if (v <= vmax && v >= vmin) {
                pp->weight[n] = v * 10.0;

                p = (pgrid->Xe - pgrid->Xs) * ix / (Xdots * 2.0);
                pp->x[n] = pgrid->Xs + ((pgrid->Xe - pgrid->Xs) / 4.0) + p;

                p = (pgrid->Ye - pgrid->Ys) * iy / (Ydots * 2.0);
                pp->y[n] = pgrid->Ys + ((pgrid->Ye - pgrid->Ys) / 4.0) + p;

                pp->vx[n] = pp->vy[n] = 0.0; // at start particles are still

                n++;
                if (n >= np) break;
           }
        }
        if (n >= np) break;
    }

    //if idx == 0 && idy == 0, to execute the print only once
    if ((blockIdx.x * blockDim.x + threadIdx.x == 0) && (blockIdx.y * blockDim.y + threadIdx.y == 0)) print_Population(*pp);
}


__global__ void SystemInstantEvolution(struct Population *pp, double *forces){

    struct particle p1, p2;
    int i, j;
    double f[2];
    //variables to compute force between p1 and p2
    double force, d2, dx, dy;
    static double k = 0.001, tiny = (double) 1.0 / (double) 1000000.0;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stridex = gridDim.x * blockDim.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int stridey = gridDim.y * blockDim.y;

    for (i = idx; i < pp->np; i+=stridex) {
        forces[index2D(0, i, 2)] = 0.0;
        forces[index2D(1, i, 2)] = 0.0;

	p1.weight = pp->weight[i];
	p1.x = pp->x[i];
	p1.y = pp->y[i];
	p1.vx = pp->vx[i];
	p1.vy = pp->vy[i];
	
	for (j = idy; j < pp->np; j+=stridey) {
            if (j != i) {
		p2.weight = pp->weight[j];
		p2.x = pp->x[j];
		p2.y = pp->y[j];
		p2.vx = pp->vx[j];
		p2.vy = pp->vy[j];
		
		//force computation between p1 and p2
		dx = p2.x - p1.x;
		dy = p2.y - p1.y;
		d2 = dx * dx + dy * dy;  // what if particles get in touch? Simply avoid the case
		if (d2 < tiny) d2 = tiny;
		force = (k * p1.weight * p2.weight) / d2;
		f[0] = force * dx / sqrt(d2);
		f[1] = force * dy / sqrt(d2);

		forces[index2D(0, i, 2)] = forces[index2D(0, i, 2)] + f[0];
                forces[index2D(1, i, 2)] = forces[index2D(1, i, 2)] + f[1];
                forces[index2D(1, i, 2)] = forces[index2D(1, i, 2)] + f[1];
                forces[index2D(0, i, 2)] = forces[index2D(0, i, 2)] + f[0];
                forces[index2D(1, i, 2)] = forces[index2D(1, i, 2)] + f[1];
	    }
        }
    }
}


void SystemEvolution(struct i2dGrid *pgrid, struct Population *pp, int mxiter, double timebit, double min, double max) {
    int t;

    int N = (pgrid->EX) * (pgrid-> EY);
    double *g_forces;
    cudaMalloc(&g_forces, 2 * pp->np * sizeof(double));

    // TODO ?
    dim3 threads_per_block (2, 2, 1); // 32 * 32 = 1024, maximum number of threads per block
    //dim3 threads_per_block (32, 32, 1); // 32 * 32 = 1024, maximum number of threads per block

    dim3 number_of_blocks (2, 2, 1); // (2 * 80) < 65535, maximum number of blocks per grid dimension
    //dim3 number_of_blocks (2 * num_SMs, 2 * num_SMs, 1); // (2 * 80) < 65535, maximum number of blocks per grid dimension

    dim3 threads_per_block_uni (2, 1, 1); // 32 * 32 = 1024, maximum number of threads per block
    //dim3 threads_per_block (32, 1, 1); // 32 * 32 = 1024, maximum number of threads per block

    dim3 number_of_blocks_uni (2, 1, 1); // (2 * 80) < 65535, maximum number of blocks per grid dimension
    //dim3 number_of_blocks (2 * num_SMs, 1, 1); // (2 * 80) < 65535, maximum number of blocks per grid dimension

    // compute forces acting on each particle step by step
    for (t = 0; t < mxiter; t++) {
        fprintf(stdout, "Step %d of %d\n", t, mxiter);
        ParticleScreen(pgrid, pp, t, min, max);
        // DumpPopulation call frequency may be changed
        if (t % 4 == 0) DumpPopulation(*pp, t);
        ParticleStats(pp, t);

	Population * pp_dev;
	double * temp_dev;
    cudaMalloc(&pp_dev, sizeof(struct Population));
    cudaMemcpy(&(pp_dev->np), &(pp->np), sizeof(int), cudaMemcpyHostToDevice);	

	cudaMalloc(&temp_dev, N * sizeof(double));
	cudaMemcpy(temp_dev, pp->weight, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(&(pp_dev->weight), &temp_dev, sizeof(double *), cudaMemcpyHostToDevice);

	temp_dev = NULL;
	cudaMalloc(&temp_dev, N * sizeof(double));
	cudaMemcpy(temp_dev, pp->x, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(&(pp_dev->x), &temp_dev, sizeof(double *), cudaMemcpyHostToDevice);

	temp_dev = NULL;
	cudaMalloc(&temp_dev, N * sizeof(double));
	cudaMemcpy(temp_dev, pp->y, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(&(pp_dev->y), &temp_dev, sizeof(double *), cudaMemcpyHostToDevice);

	temp_dev = NULL;
	cudaMalloc(&temp_dev, N * sizeof(double));
	cudaMemcpy(temp_dev, pp->vx, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(&(pp_dev->vx), &temp_dev, sizeof(double *), cudaMemcpyHostToDevice);
	
	temp_dev = NULL;
	cudaMalloc(&temp_dev, N * sizeof(double));
	cudaMemcpy(temp_dev, pp->vy, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(&(pp_dev->vy), &temp_dev, sizeof(double *), cudaMemcpyHostToDevice);
    temp_dev = NULL;

	SystemInstantEvolution<<<number_of_blocks, threads_per_block>>>(pp_dev, g_forces);

    cudaDeviceSynchronize();

	ComptPopulation<<<number_of_blocks_uni, threads_per_block_uni>>>(pp_dev, g_forces, timebit);
	cudaDeviceSynchronize();

    cudaFree(g_forces);
	cudaFree(pp_dev);
    cudaFree(temp_dev);
    }
}   // end SystemEvolution


__global__ void InitializeEmptyGridInt(struct i2dGrid *pgrid){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stridex = gridDim.x * blockDim.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int stridey = gridDim.y * blockDim.y;

    for (int ix = idx; ix < pgrid->EX; ix+=stridex) {
        for (int iy = idy; iy < pgrid->EY; iy+=stridey) {
            pgrid->Values[index2D(ix, iy, pgrid->EX)] = 0;
	}
    }
}

void ParticleScreen(struct i2dGrid *pgrid, struct Population * pp, int step, double rmin, double rmax) {
    // Distribute a particle population in a grid for visualization purposes

    int ix, iy, Xdots, Ydots;
    int n, wp;
    int static vmin, vmax;
    double Dx, Dy, wint, wv;
    char name[40];

    Xdots = pgrid->EX;
    Ydots = pgrid->EY;

    int N = (pgrid->EX) * (pgrid->EY);
    dim3 threads_per_block (2, 2, 1); // TODO: set dimensions of x and y dimensions
    dim3 number_of_blocks (2, 2, 1); // (2 * 80) < 65535, maximum number of blocks per grid dimension

    //initialization of particlegrid.values in device
    int * temp_dev;
    cudaMalloc(&temp_dev, N * sizeof(int));

    i2dGrid * pgrid_dev;
    int * v_dev;
    cudaMalloc(&pgrid_dev, sizeof(struct i2dGrid));
    cudaMalloc(&v_dev, N * sizeof(int));
    cudaMemcpy(pgrid_dev, pgrid, sizeof(struct i2dGrid), cudaMemcpyHostToDevice);
    cudaMemcpy(v_dev, pgrid->Values, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&(pgrid_dev->Values), &v_dev, sizeof(int *), cudaMemcpyHostToDevice);

    //input should be pointer to i2grid and not to int[], either change the signature or add memcpy to new i2grid * pointer
    InitializeEmptyGridInt<<<number_of_blocks, threads_per_block>>>(pgrid_dev);
    cudaDeviceSynchronize();

    temp_dev = NULL;
    cudaMalloc(&temp_dev, N * sizeof(int));
    cudaMemcpy(&temp_dev, &(pgrid_dev->Values), sizeof(int *), cudaMemcpyDeviceToHost);
    cudaMemcpy(pgrid->Values, temp_dev, N * sizeof(int), cudaMemcpyDeviceToHost);

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

    cudaFree(temp_dev);
    cudaFree(pgrid_dev);
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
    // TODO doesn't v_dev need cudaMalloc()?
    cudaMemcpy(v_dev, idata, N * sizeof(int), cudaMemcpyHostToDevice);

    // Run kernel with optimal number of threads
    int n_threads = sqrt(N);  // minimum for x + N/x
    if (n_threads > SHARED_MEM_MAX_THREADS)
        n_threads = SHARED_MEM_MAX_THREADS;

    MinMaxIntVal<<<1, n_threads>>>(N, v_dev, rmin_dev, rmax_dev);  // shared memory only works in the same block
    cudaDeviceSynchronize();

    cudaMemcpy(&rmin, rmin_dev, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&rmax, rmax_dev, sizeof(int), cudaMemcpyDeviceToHost);

    /*
       rmin = MinIntVal(s1 * s2, idata);
    rmax = MaxIntVal(s1 * s2, idata);*/ //TODO: MinIntVal and MaxIntVal should be made device functions
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
#include <time.h>
    time_t t0, t1;

    time(&t0);
    fprintf(stdout, "Starting at: %s", asctime(localtime(&t0)));

    InitGrid("Particles.inp");

    int * values_dev;
    cudaMalloc(&GenFieldGrid_dev, sizeof(struct i2dGrid));

    int N = GenFieldGrid.EX * GenFieldGrid.EY;

    //allocation of device variables to be used in kernels

    cudaMalloc(&values_dev, N * sizeof(int));
    cudaMalloc(&TimeBit_dev, sizeof(double));

    //copying memory from host to device
    cudaMemcpy(GenFieldGrid_dev, &GenFieldGrid, sizeof(struct i2dGrid), cudaMemcpyHostToDevice);

    //get number of multiprocessors to use in allocation of grids to further improve the performance
    int deviceId;
    int num_SMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, deviceId);
    printf("GeneratingField...\n");

    // GenFieldGrid

    // TODO ?
    dim3 threads_per_block (2, 2, 1); // 32 * 32 = 1024, maximum number of threads per block
    //dim3 threads_per_block (32, 32, 1); // 32 * 32 = 1024, maximum number of threads per block

    dim3 number_of_blocks (2, 2, 1); // (2 * 80) < 65535, maximum number of blocks per grid dimension
    //dim3 number_of_blocks (2 * num_SMs, 2 * num_SMs, 1); // (2 * 80) < 65535, maximum number of blocks per grid dimension

    GeneratingField <<<number_of_blocks, threads_per_block>>> (GenFieldGrid_dev, MaxIters, values_dev);

    cudaDeviceSynchronize(); // Wait for the GPU as all the steps in main need to be sequential

    //cudaMemcpy(GenFieldGrid.Values, values_dev, N * sizeof(int), cudaMemcpyDeviceToHost);  // TODO ?
    cudaMemcpy(TimeBit_dev, &TimeBit, sizeof(double), cudaMemcpyHostToDevice);

    // Particle population initialization

    Population * Particles_dev;
    cudaMalloc(&Particles_dev, sizeof(struct Population));

    // MIN-MAX
    // Initialize containers and device copies
    int vmin, vmax;
    int *vmin_dev, *vmax_dev;
    cudaMalloc(&vmin_dev, sizeof(int));
    cudaMalloc(&vmax_dev, sizeof(int));

    // Run kernel with optimal number of threads
    int n_threads = sqrt(N);  // minimum for x + N/x
    if (n_threads > SHARED_MEM_MAX_THREADS)
        n_threads = SHARED_MEM_MAX_THREADS;

    MinMaxIntVal<<<1, n_threads>>>(N, values_dev, vmin_dev, vmax_dev);  // shared memory only works in the same block
    cudaDeviceSynchronize();

    cudaMemcpy(&vmin, vmin_dev, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&vmax, vmax_dev, sizeof(int), cudaMemcpyDeviceToHost);

    // Allocating ParticleGrid on device
    cudaMalloc(&ParticleGrid_dev, sizeof(struct i2dGrid));
    cudaMemcpy(ParticleGrid_dev, &ParticleGrid, sizeof(struct i2dGrid), cudaMemcpyHostToDevice);

    CountPopulation<<<1, n_threads>>>(N, values_dev, &(Particles_dev->np), vmin, vmax);  // shared memory only works in the same block
    cudaDeviceSynchronize();

    cudaMemcpy(&(Particles.np), &(Particles_dev->np), sizeof(int), cudaMemcpyDeviceToHost);

    // Allocating ParticleGrid on device // TODO why is this repeated?
    cudaMalloc(&ParticleGrid_dev, sizeof(struct i2dGrid));
    cudaMemcpy(ParticleGrid_dev, &ParticleGrid, sizeof(struct i2dGrid), cudaMemcpyHostToDevice);

    // Allocating Particles on device
    double * temp_dev;
    int population_count = Particles.np;
    cudaMalloc(&temp_dev, population_count * sizeof(double));
    cudaMemcpy(&(Particles_dev->weight), &temp_dev, sizeof(double *), cudaMemcpyHostToDevice);

    temp_dev = NULL;
    cudaMalloc(&temp_dev, population_count * sizeof(double));
    cudaMemcpy(&(Particles_dev->x), &temp_dev, sizeof(double *), cudaMemcpyHostToDevice);

    temp_dev = NULL;
    cudaMalloc(&temp_dev, population_count * sizeof(double));
    cudaMemcpy(&(Particles_dev->y), &temp_dev, sizeof(double *), cudaMemcpyHostToDevice);

    temp_dev = NULL;
    cudaMalloc(&temp_dev, population_count * sizeof(double));
    cudaMemcpy(&(Particles_dev->vx), &temp_dev, sizeof(double *), cudaMemcpyHostToDevice);

    temp_dev = NULL;
    cudaMalloc(&temp_dev, population_count * sizeof(double));
    cudaMemcpy(&(Particles_dev->vy), &temp_dev, sizeof(double *), cudaMemcpyHostToDevice);

    // Allocating Particles on host

    Particles.weight = (double *) malloc(Particles.np * sizeof(double));
    Particles.x = (double *) malloc(Particles.np * sizeof(double));
    Particles.y = (double *) malloc(Particles.np * sizeof(double));
    Particles.vx = (double *) malloc(Particles.np * sizeof(double));
    Particles.vy = (double *) malloc(Particles.np * sizeof(double));

    printf("ParticleGeneration...\n");

    ParticleGeneration <<<number_of_blocks, threads_per_block>>> (GenFieldGrid_dev, ParticleGrid_dev, Particles_dev, values_dev, vmin, vmax);

    cudaDeviceSynchronize(); // Wait for the GPU as all the steps in main need to be sequential

    cudaError_t error = cudaGetLastError();

    // Compute evolution of the particle population

    temp_dev = NULL;
    cudaMalloc(&temp_dev, population_count * sizeof(double));
    cudaMemcpy(&temp_dev, &(Particles_dev->weight), sizeof(double *), cudaMemcpyDeviceToHost);
    cudaMemcpy(Particles.weight, temp_dev, population_count * sizeof(double), cudaMemcpyDeviceToHost);

    temp_dev = NULL;
    cudaMalloc(&temp_dev, population_count * sizeof(double));
    cudaMemcpy(&temp_dev, &(Particles_dev->x), sizeof(double *), cudaMemcpyDeviceToHost);
    cudaMemcpy(Particles.x, temp_dev, population_count * sizeof(double), cudaMemcpyDeviceToHost);

    temp_dev = NULL;
    cudaMalloc(&temp_dev, population_count * sizeof(double));
    cudaMemcpy(&temp_dev, &(Particles_dev->y), sizeof(double *), cudaMemcpyDeviceToHost);
    cudaMemcpy(Particles.y, temp_dev, population_count * sizeof(double), cudaMemcpyDeviceToHost);

    temp_dev = NULL;
    cudaMalloc(&temp_dev, population_count * sizeof(double));
    cudaMemcpy(&temp_dev, &(Particles_dev->vx), sizeof(double *), cudaMemcpyDeviceToHost);
    cudaMemcpy(Particles.vx, temp_dev, population_count * sizeof(double), cudaMemcpyDeviceToHost);

    temp_dev = NULL;
    cudaMalloc(&temp_dev, population_count * sizeof(double));
    cudaMemcpy(&temp_dev, &(Particles_dev->vy), sizeof(double *), cudaMemcpyDeviceToHost);
    cudaMemcpy(Particles.vy, temp_dev, population_count * sizeof(double), cudaMemcpyDeviceToHost);

    // MIN-MAX
    // Initialize containers and device copies
    double *rmin_dev, *rmax_dev;
    double rmin, rmax;

    cudaMalloc(&rmin_dev, sizeof(double));
    cudaMalloc(&rmax_dev, sizeof(double));

    // Run kernel with optimal number of threads
    n_threads = sqrt(Particles.np);  // minimum for x + N/x
    if (n_threads > SHARED_MEM_MAX_THREADS)
	    n_threads = SHARED_MEM_MAX_THREADS;

    double * weight_dev;

    cudaMalloc(&weight_dev, Particles.np * sizeof(double));
    cudaMemcpy(weight_dev, (Particles.weight), Particles.np * sizeof(double), cudaMemcpyHostToDevice);

    MinMaxDoubleVal<<<1, n_threads>>>(Particles.np, weight_dev, rmin_dev, rmax_dev); // shared memory only works in the same block

    cudaDeviceSynchronize();

    cudaMemcpy(&rmin, rmin_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&rmax, rmax_dev, sizeof(double), cudaMemcpyDeviceToHost);

    printf("SystemEvolution...\n");
    SystemEvolution (&ParticleGrid, &Particles, MaxSteps, TimeBit, rmin, rmax);

    cudaFree(GenFieldGrid_dev);
    cudaFree(values_dev);
    cudaFree(TimeBit_dev);
    cudaFree(Particles_dev);
    cudaFree(vmin_dev);
    cudaFree(vmax_dev);
    cudaFree(ParticleGrid_dev);
    cudaFree(temp_dev);
    cudaFree(rmin_dev);
    cudaFree(rmax_dev);
    cudaFree(rmin_dev);
    cudaFree(rmax_dev);
    cudaFree(weight_dev);

    time(&t1);
    fprintf(stdout, "Ending   at: %s", asctime(localtime(&t1)));
    fprintf(stdout, "Computations ended in %lf seconds\n", difftime(t1, t0));

    fprintf(stdout, "End of program!\n");

    return (0);
}  // end FinalApplication
