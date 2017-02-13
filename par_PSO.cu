#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

// ____  Experiment inputs _____
#define N 10   // Number of variables
#define F_LINE pow(my_var,2) // For Sphere function x_i^2
#define INIT_VALUE 0
#define POPULATION 128 // Population
#define UB 10 // Upper bound
#define LB -10 // Lower bound
#define N_ITERS 1000
#define N_EXPERIMENTS 1
// _____________________________

#if ((N % 32) == 0)
  #define THREADS N
  #define WARP_GAP 0
#else
  #define THREADS (N/32 + 1)*32
  #define WARP_GAP (N/32 + 1)*32 - N
#endif
#define W 0.6
#define PHI_P 1.6
#define PHI_G 1.6
#define N_WARPS N/WARP_SIZE
#define F_KEY FUNCT
#define WARP_SIZE 32
#define N_WARPS N/WARP_SIZE
__device__ int iMin = 0;
__device__ int fitMin = 9999999;

// Prints an array of floats from Device
void fcudaPrint(float * array, int elements, int n_jump){
  float * aux;
  aux = (float *) malloc(elements * sizeof(float));
  cudaMemcpy(aux, array, elements * sizeof(float), cudaMemcpyDeviceToHost);

  int i;
  for (i = 0; i < elements; i++){
      if ((i % n_jump) == 0 ) {printf("\n");}
      printf("%.5f ", aux[i]);
  }
  free(aux);
  aux = NULL;
}

// Prints an array of ints from Device
void icudaPrint(int * array, int elements, int n_jump){
  int * aux;
  aux = (int *) malloc(elements * sizeof(int));
  cudaMemcpy(aux, array, elements * sizeof(int), cudaMemcpyDeviceToHost);

  int i;
  for (i = 0; i < elements; i++){
      if ((i % n_jump) == 0 ) {printf("\n");}
      printf("%d ", aux[i]);
  }
  free(aux);
  aux = NULL;
}


__global__ void init_states(unsigned int seed, curandState_t * states) {
  /* we have to initialize the state */
  curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}

__global__ void U_01(float * arr, curandState_t * states){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  arr[id] = curand_uniform(&states[id]);
}

__global__ void N_01(float * arr, curandState_t * states){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  arr[id] = curand_normal(&states[id])/1;
}

// Computes random integers between 0 and n
__global__ void irand(int n, int * arr, curandState_t * states){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  arr[id] = curand(&states[id]) % n;
}

__global__ void FUNCT(float * x, float * evals){
  // Launch 1 block of N threads for each element fo the population: <<<POPULATION, N>>>
  // variable sum is shared by all threads in each block
  __shared__ float sum;
  float value;
  int phantom_id = blockIdx.x * blockDim.x + threadIdx.x; // Id with gaps included
  int real_id = phantom_id - blockIdx.x * ((int)WARP_GAP); // Id withouth gaps included
  if (threadIdx.x == 0) {
    sum = INIT_VALUE; //
  }
  // The main operation
  if (threadIdx.x < N){
    float my_var = x[real_id];
    value = F_LINE;
  } else {
    value = 0;
  }
  // A little warp reduction in order to reduce the amount of atomic operations
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value += __shfl_down(value, offset);
  }
  // The first thread of each warp adds its value
  if ((threadIdx.x & 31) == 0){
    atomicAdd(&sum, value);
  }
  // Thread synchronization, because this is not a warp operation
  __syncthreads();
  // Only one thread writes the result of this block-bee
  if (threadIdx.x == 0){
      evals[blockIdx.x] = sum;
  }
}

// Sets up the initial population
__global__ void init_pos(float * pos, float * rand_uniform){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  pos[id] = LB + rand_uniform[id] * (UB - LB);
}

// Sets up the initial velocity
__global__ void init_vel(float * vel, float * rand_A, float * rand_B){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  vel[id] = LB + (rand_B[id] -  rand_A[id]) * (UB - LB);
}

// Updates the min array with the actual position of each particle
__global__ void update_mins(float * pos, float * min, float * evals, float * min_evals){
  // Needed to lauch as many blocks as particles, with N threads per block
  //int id = blockIdx.x * blockDim.x + threadIdx.x;
  int phantom_id = blockIdx.x * blockDim.x + threadIdx.x; // Id with gaps included
  int real_id = phantom_id - blockIdx.x * ((int)WARP_GAP); // Id withouth gaps included
  int j = blockIdx.x;
  // We're using more threads than needed, but we maximize the GPU usage
  if (threadIdx.x < N){
    if (evals[j] < min_evals[j]){
      if (threadIdx.x == 0){min_evals[j] = evals[j];}
      min[real_id] = pos[real_id];
      }
    }
  }

// Operación atómica que escribe el mínimo de un índice
__device__ float atomicMinIndex(float * array, int * address, int val){
  int lo_que_tengo, lo_que_tenia;
  lo_que_tengo = * address;
  while (array[val] < array[lo_que_tengo]){
    lo_que_tenia = lo_que_tengo;
    lo_que_tengo = atomicCAS(address, lo_que_tenia, val);
  }
  return lo_que_tengo;
}

__global__  void arrayReduction(float * array){
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int thisThreadId = id;
  float value = array[id];
  int gap, id2;
  float value2;
  for (gap = WARP_SIZE/2; gap > 0; gap >>= 1){
    id2 = __shfl_down(id, gap);
    value2 = __shfl_down(value, gap);
    if (value2 < value){
      value = value2;
      id = id2;
    }
  }
  if (((thisThreadId & (WARP_SIZE - 1)) == 0)){
    atomicMinIndex(array, &iMin, id);
  }
}

// Updates the velocity & pos
__global__ void update_vel_pos(float * vel, float * pos, float * min, float * ru01, float * ru02){
  // It's necessary launch as many blocks of N threads as particles
  //int real_id = blockIdx.x * blockDim.x + threadIdx.x;
  int i = threadIdx.x;

  int phantom_id = blockIdx.x * blockDim.x + threadIdx.x; // Id with gaps included
  int real_id = phantom_id - blockIdx.x * ((int)WARP_GAP); // Id withouth gaps included

  if (threadIdx.x < N){
    // Update speed
    //vel[id] = W * vel[id] + PHI_P * ru01[id] * (min[id] - pos[id]) +  PHI_G * ru02[id] * (min[iMin*N + i] - pos[id]);
    vel[real_id] = __fmaf_rd(W, vel[real_id], __fmul_rd(__fmul_rd(PHI_P, ru01[real_id]), __fsub_rd(min[real_id], pos[real_id])) +  __fmul_rd(__fmul_rd(PHI_G, ru02[real_id]), __fsub_rd(min[iMin*N + i], pos[real_id])));
    // Update position
    //pos[id] = pos[id] + vel[id];
    pos[real_id] = __fadd_rd(pos[real_id], vel[real_id]);

  }
}

// Updates the velocity & pos
__global__ void PSO_step(float * vel, float * pos, float * min, float * evals, float * min_evals, float * ru01){
  // It's necessary launch as many blocks of N threads as particles
  //int real_id = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float sum;
  int phantom_id = blockIdx.x * blockDim.x + threadIdx.x; // Id with gaps included
  int real_id = phantom_id - blockIdx.x * ((int)WARP_GAP); // Id withouth gaps included
  float my_var, my_vel;
  my_var = pos[real_id];
  my_vel = vel[real_id];
  if (threadIdx.x == 0) {sum = INIT_VALUE;}
  int i = threadIdx.x;
  float value;
  // --------- Updating pos and speed
  if (threadIdx.x < N){
    // Update speed
    my_vel = __fmaf_rd(W, my_vel, __fmul_rd(__fmul_rd(PHI_P, ru01[real_id]), __fsub_rd(min[real_id], my_var)) +  __fmul_rd(__fmul_rd(PHI_G, ru01[N*POPULATION - real_id]), __fsub_rd(min[iMin*N + i], my_var)));
    vel[real_id] = my_vel; // Saving it in global mem...
    // Update position
    my_var = __fadd_rd(my_var, my_vel);
    pos[real_id] = my_var; // Saving it in global mem...
  }
  __syncthreads();
  // --------- Evaluating the function
  if (threadIdx.x < N){
    value = F_LINE;
  } else {
    value = 0;
  }
  // A little warp reduction in order to reduce the amount of atomic operations
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value += __shfl_down(value, offset);
  }
  // The first thread of each warp adds its value
  if ((threadIdx.x & 31) == 0){
    atomicAdd(&sum, value);
    //printf("-->> Thread %d sums %f to %d bee\n", threadIdx.x, value, blockIdx.x);
  }
  // Thread synchronization, because this is not a warp operation
  __syncthreads();
  // Only one thread writes the result of this block-bee
  if (threadIdx.x == 0){
      //evals[blockIdx.x] = fitness(sum);
      evals[blockIdx.x] = sum; // From here, sum is the complete evaluation
      //printf("--%f, bee: %d\n", sum, blockIdx.x);
  }

  __syncthreads();

  int j = blockIdx.x;
  // ------------- Updating mins
  if (threadIdx.x < N){
    //if (evals[j] < min_evals[j]){
    if (sum < min_evals[j]){
      if (threadIdx.x == 0){min_evals[j] = sum;}
      min[real_id] = my_var;
      //printf("done %d \n", i);
    }
  }

}

__global__ void print_best_pos(float * pos){
  int i;
  for (i = 0; i < N; i++){
    printf("%f ", pos[iMin * N + i]);
  }
  printf("\n");
}

__global__ void print_best_val(float * part_min_evals){
  printf("[Solution] Value: %f\n", part_min_evals[iMin]);
}

// Return the floating mean of the n first elements from arr
float fmean(float * arr, int n){
  float ssum = 0;
  int i;
  for (i = 0; i < n; i++){
    ssum += arr[i];
  }
  return ssum/n;
}

// Prints an array of floats
void fMatrixPrint(float * array, int elements, int n_jump){
  int i;
  for (i = 0; i < elements; i++){
      if ((i % n_jump) == 0 ) {printf("\n");}
      printf("%f ", array[i]);
  }
}

int main(void){
  // States
  curandState_t * states;
  cudaMalloc((void**) &states, N * POPULATION * sizeof(curandState_t));
  init_states<<<N*POPULATION,1>>>(time(0), states);

  // The random things
  float * rand_float_A, * rand_float_B;
  cudaMalloc((void **) &rand_float_A, N * POPULATION * sizeof(float));
  cudaMalloc((void **) &rand_float_B, N * POPULATION * sizeof(float));

  // The data structures
  float * particles, * evals;
  float * part_min, * part_min_evals;
  float * vel;

  cudaMalloc((void **) &particles, N * POPULATION * sizeof(float));
  cudaMalloc((void **) &evals, POPULATION * sizeof(float));
  cudaMalloc((void **) &part_min, N * POPULATION * sizeof(float));
  cudaMalloc((void **) &part_min_evals, POPULATION * sizeof(float));
  cudaMalloc((void **) &vel, N * POPULATION * sizeof(float));
  U_01<<<N, POPULATION>>>(rand_float_A, states);
  U_01<<<N, POPULATION>>>(rand_float_B, states);
  init_pos<<<POPULATION, N>>>(particles, rand_float_A);
  init_pos<<<POPULATION, N>>>(part_min, rand_float_B);
  init_vel<<<POPULATION, N>>>(vel, rand_float_A, rand_float_B);
  F_KEY<<<POPULATION,THREADS>>>(particles, evals);
  F_KEY<<<POPULATION,THREADS>>>(part_min, part_min_evals);
  cudaDeviceSynchronize(); clock_t start, end;
  double cpu_time_used;
  float timings[N_EXPERIMENTS];
  int experiment;
  for (experiment = 0; experiment < N_EXPERIMENTS; experiment ++){
    start = clock();
    int iter;
    for (iter = 0; iter < N_ITERS; iter++){
      U_01<<<N, POPULATION>>>(rand_float_A, states);
      PSO_step<<<POPULATION,THREADS>>>(vel, particles, part_min, evals, part_min_evals, rand_float_A);
      // Look for global best
      arrayReduction<<<1,POPULATION>>>(part_min_evals);
    }
  cudaDeviceSynchronize();
  end = clock();
  cpu_time_used = ((double) (end - start))/ CLOCKS_PER_SEC;
  timings[experiment] = (float) cpu_time_used;
  }
  float time_mean = fmean(timings, N_EXPERIMENTS);
  printf("\n[Info] Iterations : %d\n[Info] Experiments: %d\n[Info] Variables  : %d\n[Info] Mean time  : %f\n", N_ITERS, N_EXPERIMENTS, N, time_mean);
  print_best_val<<<1,1>>>(part_min_evals);
  cudaDeviceSynchronize();
  printf("[Solution] Location: ");
  print_best_pos<<<1,1>>>(part_min);
  cudaDeviceSynchronize();
  printf("\n");

  return 0;
}
