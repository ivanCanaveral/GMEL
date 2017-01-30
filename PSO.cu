#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

// Function provided
#define F_LINE pow(my_pos,2)
#define INIT_VALUE 0
#define UB 10 // Upper bound
#define LB -10 // Lower bound

// Experiment configuration
#define N 32   // Number of variables
#define POPULATION 128 // Population
#define N_ITERS 1000 // Max number of iterations
#define N_EXPERIMENTS 100 // Number of experiments

// PSO Parameters
#define W 0.6
#define PHI_P 1.6
#define PHI_G 1.6


#if ((N % 32) == 0)
  #define THREADS N
  #define WARP_GAP 0
#else
  #define THREADS (N/32 + 1)*32
  #define WARP_GAP (N/32 + 1)*32 - N
#endif
#define N_WARPS N/WARP_SIZE
#define F_KEY SPH
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

// Function to minimize : Sphere
__global__ void SPH(float * x, float * evals){
  // Launch 1 block of N threads for each element fo the population: <<<POPULATION, N>>>

  // variable sum is shared by all threads in each block
  __shared__ float sum;
  //__shared__ float prod;
  //__shared__ float sum2;

  float value;
  int phantom_id = blockIdx.x * blockDim.x + threadIdx.x; // Id with gaps included
  int real_id = phantom_id - blockIdx.x * ((int)WARP_GAP); // Id withouth gaps included

  // Init the sum value to 0
  if (threadIdx.x == 0) {
    //sum = 0; // Sphere SPH
    sum = 10*N; // Rastringin RTG
  }
  // The main operation
  //value = __fmul_rd(__fsub_rd(x[i],1),__fsub_rd(x[i],1));
  //__syncthreads();

  if (threadIdx.x < N){
    //value = pow(x[real_id],2); // Sphere SPH
    value = pow(x[real_id],2) - __fmul_rd(10, __cosf(2 * M_PI * x[real_id])); // Rastringin RTG
  } else {
    value = 0;
  }
  //printf("8====D %f : %d < %d\n", value, threadIdx.x, N);
  //__syncthreads();

  // A little warp reduction in order to reduce the amount of atomic operations
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value += __shfl_down(value, offset);
  }
  //printf("----%f\n", value);
  //if ((threadIdx.x & 31) == 0){
  //  printf("----%f\n", value);
  //  printf("-->> Thread %d sums %f to %d bee\n", threadIdx.x, value, blockIdx.x);
  //}

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
      evals[blockIdx.x] = sum;
      //printf("--%f, bee: %d\n", sum, blockIdx.x);
  }
}

// Sets up the initial population
__global__ void init_pos(float * pos, float * rand_uniform){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  //pos[id] = LB + rand_uniform[id] * (UB - LB);
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
  //printf("%d", i);
  if (threadIdx.x < N){
    if (evals[j] < min_evals[j]){
      if (threadIdx.x == 0){min_evals[j] = evals[j];}
      min[real_id] = pos[real_id];
      //printf("done %d \n", i);
      }
    }
  }

// Operación atómica que escribe el mínimo de un índice
__device__ float atomicMinIndex(float * array, int * address, int val){
  int lo_que_tengo, lo_que_tenia;
  lo_que_tengo = * address;
  //printf("Esto : %f debe ser menor que esto %f\n", array[val], array[lo_que_tengo]);
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

  float my_pos, my_vel;
  my_pos = pos[real_id];
  my_vel = vel[real_id];

  if (threadIdx.x == 0) {sum = INIT_VALUE;}

  int i = threadIdx.x;

  float value;



  // --------- Updating pos and speed
  if (threadIdx.x < N){
    // Update speed
    my_vel = __fmaf_rd(W, my_vel, __fmul_rd(__fmul_rd(PHI_P, ru01[real_id]), __fsub_rd(min[real_id], my_pos)) +  __fmul_rd(__fmul_rd(PHI_G, ru01[N*POPULATION - real_id]), __fsub_rd(min[iMin*N + i], my_pos)));
    vel[real_id] = my_vel; // Saving it in global mem...
    // Update position
    my_pos = __fadd_rd(my_pos, my_vel);
    pos[real_id] = my_pos; // Saving it in global mem...
  }

  __syncthreads();

  // --------- Evaluating the function

  if (threadIdx.x < N){
    value = F_LINE;
  } else {
    value = 0;
  }
  //printf("8====D %f : %d < %d\n", value, threadIdx.x, N);
  //__syncthreads();

  // A little warp reduction in order to reduce the amount of atomic operations
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value += __shfl_down(value, offset);
  }
  //printf("----%f\n", value);
  //if ((threadIdx.x & 31) == 0){
  //  printf("----%f\n", value);
  //  printf("-->> Thread %d sums %f to %d bee\n", threadIdx.x, value, blockIdx.x);
  //}

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
      min[real_id] = my_pos;
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
  printf("FIT: %f\n", part_min_evals[iMin]);
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

  //printf("Test: Nmod32 = %d, THREADS = %d, valor de la prueba %d\n", N%32, THREADS, (N % 32) == 0);
  //printf("Valor de la variable THREADS: %d\n", THREADS);

  //printf("Valor del GAP: %d\n", WARP_GAP);
  // States
  curandState_t * states;
  cudaMalloc((void**) &states, N * POPULATION * sizeof(curandState_t));
  init_states<<<N*POPULATION,1>>>(time(0), states);

  // The random things
  float * rand_float_A, * rand_float_B;
  //int * rand_integer;
  cudaMalloc((void **) &rand_float_A, N * POPULATION * sizeof(float));
  cudaMalloc((void **) &rand_float_B, N * POPULATION * sizeof(float));
  //cudaMalloc((void **) &rand_integer, N * POPULATION * sizeof(int));

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
  //fcudaPrint(rand_float_A, N*POPULATION, N);
  //printf("\n");
  //fcudaPrint(rand_float_B, N*POPULATION, N);
  //irand<<<N,POPULATION>>>(50, rand_integer, states);

  //fcudaPrint(rand_float, N*POPULATION, N);

  // -- Init ----
  init_pos<<<POPULATION, N>>>(particles, rand_float_A);
  init_pos<<<POPULATION, N>>>(part_min, rand_float_B);
  //fcudaPrint(particles, N*POPULATION, N);
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
      // Rands
      U_01<<<N, POPULATION>>>(rand_float_A, states);
      //U_01<<<N, POPULATION>>>(rand_float_B, states);
      // Update speed & pos
      ///update_vel_pos<<<POPULATION,THREADS>>>(vel, particles, part_min, rand_float_A, rand_float_B);
      PSO_step<<<POPULATION,THREADS>>>(vel, particles, part_min, evals, part_min_evals, rand_float_A);
      // Update position
      //update_pos<<<POPULATION,N>>>(particles, vel);
      // Eval function over new positions
      ///F_KEY<<<POPULATION,THREADS>>>(particles, evals);
      // Sets the particle's min
      //update_mins<<<POPULATION,THREADS>>>(particles, part_min, evals, part_min_evals);
      // Look for global best
      arrayReduction<<<1,POPULATION>>>(part_min_evals);
    }

  cudaDeviceSynchronize();
  end = clock();
  cpu_time_used = ((double) (end - start))/ CLOCKS_PER_SEC;
  
  timings[experiment] = (float) cpu_time_used;
  }
  
  float time_mean = fmean(timings, N_EXPERIMENTS);

  // [N_ITERS] [N_EXPERIMENTS] [N_VARS] [mean_time]
  printf("%d %d %d %f\n", N_ITERS, N_EXPERIMENTS, N, time_mean);
  //fcudaPrint(particles, N*POPULATION, N);
  //fcudaPrint(evals, POPULATION, POPULATION);
  //fcudaPrint(part_min, N*POPULATION, N);
  //fcudaPrint(part_min_evals, POPULATION, POPULATION);

  //update_mins<<<POPULATION,N>>>(particles, part_min, evals, part_min_evals);
  //fcudaPrint(part_min, N*POPULATION, N);
  //fcudaPrint(part_min_evals, POPULATION, POPULATION);

  //arrayReduction<<<1,POPULATION>>>(part_min_evals);
  //update_vel<<<POPULATION,N>>>(vel, particles, part_min, rand_float_A, rand_float_B);
  //update_pos<<<POPULATION,N>>>(particles, vel);
  //printf("\n");
  print_best_pos<<<1,1>>>(part_min);
  //print_best_val<<<1,1>>>(part_min_evals);
  cudaDeviceSynchronize();

  //fMatrixPrint(timings, N_EXPERIMENTS, N_EXPERIMENTS);
  
  return 0;
}
