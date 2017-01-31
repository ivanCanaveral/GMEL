/*
FUNCTION
---------------------
    SPH .   Sphere.                 Min : x_i = 1                     Bounds : [-10, 10]
    SKT .   Styblinski-Tang         Min : x_i = -2.903534             Bounds : [-5 , 5 ]
    DXP .   Dixon-Price             Min : x_i = 2^(-(2^i - 2)/(2^2))  Bounds : [-10, 10]
    RSB .   Rosenbrock              Min : x_i = 1                     Bounds : [-2.048, 2.048]
    ACK .   Ackley                  Min : x_i = 0                     Bounds : [-32.768, 32.768]
    GWK .   Griewank                Min : x_i = 0                     Bounds : [-600, 600]
    RTG .   Rastrigin               Min : x_i = 0                     Bounds : [-5.12, 5.12]
    LEV .   Levy                    Min : x_i = 1                     Bounds : [-10, 10]
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define F_KEY SPH
#define F_NAME "Sphere"

#define N 32  // Dimension
#define POPULATION 128 // Population
#define UB 10 // Upper bound
#define LB -10 // Lower bound

#define CR 0.8
#define F 0.5

#define WARP_SIZE 32
#define N_WARPS N/WARP_SIZE
#define N_ITERS 1000



__device__ int iMin = 0;

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
  arr[id] = curand_normal(&states[id])/10;
}

__global__ void irand(int n, int * arr, curandState_t * states){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  arr[id] = curand(&states[id]) % n;
}

// Set up the initial population
__global__ void init_pos(float * pos, float * rand_uniform){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  pos[id] = LB + rand_uniform[id] * (UB - LB);
}

// Fill an array with zeros
__global__ void zeros(int * arr){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  arr[id] = 0;
}


// Function to minimize : Sphere
__global__ void SPH(float * x, float * evals){
  // Launch 1 block of N threads for each element fo the population: <<<POPULATION, N>>>

  // variable sum is shared by all threads in each block
  __shared__ float sum;

  float value;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Init the sum value to 0
  if (threadIdx.x == 0) {sum = 0;}
  // The main operation
  //value = __fmul_rd(__fsub_rd(x[i],1),__fsub_rd(x[i],1));
  value = pow(x[i] - 1,2);

  // A little warp reduction in order to reduce the amount of atomic operations
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value += __shfl_down(value, offset);
  }

  // The first thread of each warp adds its value
  if ((i & 31) == 0){
    atomicAdd(&sum, value);
  }

  // Thread synchronization, because this is not a warp operation
  __syncthreads();

  // Only one thread writes the result of this block-bee
  if (threadIdx.x == 0){
      evals[blockIdx.x] = sum;
  }
}

// Function to minimize : Styblinski-Tang
__global__ void SKT(float * x, float * evals){
  // Launch 1 block of N threads for each element fo the population: <<<POPULATION, N>>>

  // variable sum is shared by all threads in each block
  __shared__ float sum;

  float value;
  int i = blockIdx.x * blockDim.x + threadIdx.x;


  if (threadIdx.x == 0) {sum = 39.166599*N;}
  // The main operation
  //value = __fadd_ru(__fsub_ru(__fmul_ru(0.5,__powf(x[i], 4)), __fmul_ru(8,__powf(x[i],2))), __fmul_ru(2.5, x[i]));
  value = 0.5 * pow(x[i], 4) - 8 * pow(x[i],2) + 2.5*x[i];

  // A little warp reduction in order to reduce the amount of atomic operations
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value += __shfl_down(value, offset);
  }

  // The first thread of each warp adds its value
  if ((i & 31) == 0){
    atomicAdd(&sum, value);
  }

  // Thread synchronization, because this is not a warp operation
  __syncthreads();

  // Only one thread writes the result of this block-bee
  if (threadIdx.x == 0){
      evals[blockIdx.x] = sum;
  }
}

// Function to minimize : Dixon-Price
__global__ void DXP(float * x, float * evals){
  // Launch 1 block of N threads for each element fo the population: <<<POPULATION, N>>>

  // variable sum is shared by all threads in each block
  __shared__ float sum;

  float value;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x == 0) {
    sum = 0;
    value = pow(x[0] - 1,2);
  } else {
    value = threadIdx.x * pow(2 *pow(x[i],2) - x[i-1] ,2);
  }

  // A little warp reduction in order to reduce the amount of atomic operations
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value += __shfl_down(value, offset);
  }

  // The first thread of each warp adds its value
  if ((i & 31) == 0){
    atomicAdd(&sum, value);
  }

  // Thread synchronization, because this is not a warp operation
  __syncthreads();

  // Only one thread writes the result of this block-bee
  if (threadIdx.x == 0){
      evals[blockIdx.x] = sum;
  }
}

// Function to minimize : Rosenbrock
__global__ void RSB(float * x, float * evals){
  // Launch 1 block of N threads for each element fo the population: <<<POPULATION, N>>>

  // variable sum is shared by all threads in each block
  __shared__ float sum;

  float value;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x == (N-1)) {
    sum = 0;
    value = 0;
  } else {
    value = 100 * pow(x[i+1] - pow(x[i],2),2) + pow(1 - x[i],2);
  }

  // A little warp reduction in order to reduce the amount of atomic operations
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value += __shfl_down(value, offset);
  }

  // The first thread of each warp adds its value
  if ((i & 31) == 0){
    atomicAdd(&sum, value);
  }

  // Thread synchronization, because this is not a warp operation
  __syncthreads();

  // Only one thread writes the result of this block
  if (threadIdx.x == 0){
      evals[blockIdx.x] = sum;
  }
}

// Function to minimize : Zakharov
__global__ void ZKV(float * x, float * evals){
  // Launch 1 block of N threads for each element fo the population: <<<POPULATION, N>>>

  // variable sum is shared by all threads in each block
  __shared__ float sum;

  float value0, value1;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Init the sum value to 0
  if (threadIdx.x == 0){
    sum = 0;
  }

  // The main operation
  value0 = pow(x[i],2);
  value1 = 0.5 * threadIdx.x * (x[i]);

  // A little warp reduction in order to reduce the amount of atomic operations
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value0 += __shfl_down(value0, offset);
    value1 += __shfl_down(value1, offset);
  }

  // The first thread of each warp adds its value

  if ((i & 31) == 0){
    atomicAdd(&sum, value0 + pow(value1, 2) + pow(value1, 4));
  }

  // Thread synchronization, because this is not a warp operation
  __syncthreads();

  // Only one thread writes the result of this block-bee
  if (threadIdx.x == 0){
      evals[blockIdx.x] = sum;
  }
}

// Function to minimize : Ackley
__global__ void ACK(float * x, float * evals){
  // Launch 1 block of N threads for each element fo the population: <<<POPULATION, N>>>

  // variable sum is shared by all threads in each block
  __shared__ float sum;

  float value0, value1;
  float addend0, addend1;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Some parameters
  float a = 20;
  float b = 0.2;
  float c = 2 * M_PI;

  // Init the sum value to 0
  if (threadIdx.x == 0){
    sum = 0;
  }

  // The main operation
  //value = __fmul_rd(__fsub_rd(x[i],1),__fsub_rd(x[i],1));
  value0 = pow(x[i], 2);
  value1 = __cosf(c * x[i]);
  // A little warp reduction in order to reduce the amount of atomic operations
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value0 += __shfl_down(value0, offset);
    value1 += __shfl_down(value1, offset);
  }

  // Compute the addends in parallel
  if (threadIdx.x == 0){
    addend0 = -a * __expf(__fmul_rd(-b, __fsqrt_rd(__frcp_rz(N) * value0)));
    //printf("%f\n", addend0);
  } else if (threadIdx.x == 1){
    addend1 = __expf(__fmul_rd(__frcp_rz(N), value1));
    //printf("%f\n", addend1);
  }

  // The first thread of each warp adds its value
  if ((i & 31) == 0){
    atomicAdd(&sum, addend0 - addend1 + a + __expf(1));
  }

  // Thread synchronization, because this is not a warp operation
  __syncthreads();

  // Only one thread writes the result of this block-bee
  if (threadIdx.x == 0){
      evals[blockIdx.x] = sum;
  }
}

// Function to minimize : Griewank
__global__ void GWK(float * x, float * evals){
  // Launch 1 block of N threads for each element fo the population: <<<POPULATION, N>>>

  // variable sum is shared by all threads in each block
  __shared__ float sum;

  float value0, value1;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Init the sum value to 0
  if (threadIdx.x == 0){
    sum = 0;
  }

  // The main operation
  //value = __fmul_rd(__fsub_rd(x[i],1),__fsub_rd(x[i],1));
  value0 = pow(x[i], 2);
  value1 = __cosf(__fdiv_rn(x[i], __fsqrt_rn(threadIdx.x + 1)));;

  // A little warp reduction in order to reduce the amount of atomic operations
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value0 += __shfl_down(value0, offset);
    value1 *= __shfl_down(value1, offset);
  }

  // The first thread of each warp adds its value
  if ((i & 31) == 0){
    atomicAdd(&sum, __fdiv_rn(value0, 4000) - value1 + 1);
  }

  // Thread synchronization, because this is not a warp operation
  __syncthreads();

  // Only one thread writes the result of this block-bee
  if (threadIdx.x == 0){
      evals[blockIdx.x] = sum;
  }
}

// Function to minimize : Rastrigin
__global__ void RTG(float * x, float * evals){
  // Launch 1 block of N threads for each element fo the population: <<<POPULATION, N>>>

  // variable sum is shared by all threads in each block
  __shared__ float sum;

  float value;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x == 0) {sum = 10*N;}

  // The main operation
  //value = __fadd_ru(__fsub_ru(__fmul_ru(0.5,__powf(x[i], 4)), __fmul_ru(8,__powf(x[i],2))), __fmul_ru(2.5, x[i]));
  value = pow(x[i],2) - 10*cos(2 * M_PI * x[i]);

  // A little warp reduction in order to reduce the amount of atomic operations
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value += __shfl_down(value, offset);
  }

  // The first thread of each warp adds its value

  if ((i & 31) == 0){
    atomicAdd(&sum, value);
  }

  // Thread synchronization, because this is not a warp operation
  __syncthreads();

  // Only one thread writes the result of this block-bee
  if (threadIdx.x == 0){
      evals[blockIdx.x] = sum;
  }
}

// Function to minimize : Levy
__global__ void LEV(float * x, float * evals){
  // Launch 1 block of N threads for each element fo the population: <<<POPULATION, N>>>

  // variable sum is shared by all threads in each block
  __shared__ float sum;

  float value;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  float w = 1 + ((x[i] - 1)/4);

  if (threadIdx.x == 0) {
    sum = 0; // Sets sum to 0
    value = pow(__sinf(__fmul_ru(M_PI, w)), 2) + pow(w - 1, 2) * (1 + 10 * pow(__sinf(M_PI * w + 1),2));
  } else if (threadIdx.x == (N-1)){
    value = pow(w - 1, 2) * (1 + pow(__sinf(2*M_PI * w),2));
  } else {
    value = pow(w - 1, 2) * (1 + 10 * pow(__sinf(M_PI * w + 1),2));
  }

  // A little warp reduction in order to reduce the amount of atomic operations
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value += __shfl_down(value, offset);
  }

  // The first thread of each warp adds its value
  if ((i & 31) == 0){
    atomicAdd(&sum, value);
  }

  // Thread synchronization, because this is not a warp operation
  __syncthreads();

  // Only one thread writes the result of this block-bee
  if (threadIdx.x == 0){
      evals[blockIdx.x] = sum;
  }
}


// Mutates the population
__global__ void mutate(float * pos, float * new_pos, int * r_gen, int * crossings, float * uniform_random){
  // One block on N threads per each member of the population
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int bug = blockIdx.x;
  int gene = threadIdx.x;
  int R = r_gen[bug]; // The random dim/gen wich will be mutated anyways
  int a = crossings[bug];
  int b = crossings[bug + 1];
  int c = crossings[bug + 2];
  float r = uniform_random[id];
  // mutation
  if (r < CR || gene == R){
    new_pos[id] = pos[a * N + gene] + F * (pos[b * N + gene] - pos[c * N + gene]);
  } else {
    new_pos[id] = pos[id];
  }
}

// Keeps the best members of the population in memory
__global__ void keep_best_members(float * pos, float * new_pos, float * evals, float * new_evals){
  // We'll need one block per member of the population (aka bug), and N threads per block
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int bug = blockIdx.x;
  int gen = threadIdx.x;

  if (new_evals[bug] < evals[bug]){
    pos[id] = new_pos[id];
    if (gen == 0) {evals[bug] = new_evals[bug];}
  }
}

// Operación atómica que escribe el mínimo de un índice
__device__ float atomicMinIndex(float * array, int * address, int val){
  int lo_que_tengo, lo_que_tenia;
  lo_que_tengo = * address;
  //printf("Esto : %f debe ser menor que esto %f\n", array[val], array[lo_que_tengo]);
  while (array[val] < array[lo_que_tengo]){
    //printf("in %f\n", array[val]);
    lo_que_tenia = lo_que_tengo;
    lo_que_tengo = atomicCAS(address, lo_que_tenia, val);
  }
  return lo_que_tengo;
}

//  In this case we don't use the influence of the global minima. In addition
//  a memen is only replaced by a better solution, so we can only use this
//  step at the end of the algorithm, unless we use this as stop criterion
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
    //printf("Intentamos colocar %d\n", id);
    atomicMinIndex(array, &iMin, id);
  }
}

void generate_random_things(curandState_t * states, float * rand_uniform, int * R, int * crossings){
  U_01<<<N, POPULATION>>>(rand_uniform, states);
  irand<<<1,POPULATION>>>(N, R, states);
  irand<<<3, POPULATION>>>(POPULATION, crossings, states);
}

// Combine the elements in the population
void combine_population(float * pos, float * new_pos, float * evals, float * new_evals, int * crossings, int * R, float * rand_uniform){
  mutate<<<POPULATION, N>>>(pos, new_pos, R, crossings, rand_uniform);

  F_KEY<<<POPULATION, N>>>(new_pos, new_evals);

  keep_best_members<<<POPULATION, N>>>(pos, new_pos, evals, new_evals);

}



__global__ void print_iMin(float * pos){

  int i;
  for (i = 0; i < N; i++){
    printf("%f ", pos[iMin * N + i]);
  }
}

__global__ void ones(float * arr){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  arr[id] = 1;
}

int main(void){

  // States
  curandState_t * states;
  cudaMalloc((void**) &states, N * POPULATION * sizeof(curandState_t));
  init_states<<<N*POPULATION,1>>>(time(0), states);

  // The random things
  float * ru01;
  int * R, * crossings;
  cudaMalloc((void **) &ru01, N * POPULATION *sizeof(float));
  cudaMalloc((void **) &R, POPULATION * sizeof(int));
  cudaMalloc((void **) &crossings, 3 * POPULATION * sizeof(int));

  U_01<<<N, POPULATION>>>(ru01, states);

  // All the stuff
  float * pos, * evals;
  float * new_pos, * new_evals;

  cudaMalloc((void **) &pos, N * POPULATION * sizeof(float));
  cudaMalloc((void **) &evals, POPULATION * sizeof(float));
  cudaMalloc((void **) &new_pos, N * POPULATION * sizeof(float));
  cudaMalloc((void **) &new_evals, POPULATION * sizeof(float));
  init_pos<<<POPULATION, N>>>(pos, ru01);
  F_KEY<<<POPULATION, N>>>(pos, evals);
  cudaDeviceSynchronize();
  clock_t start, end;
  double cpu_time_used;
  start = clock();

  //generate_random_things(states, ru01, R, crossings);
  int i;
  for (i = 0; i < N_ITERS; i ++){
    generate_random_things(states, ru01, R, crossings);
    combine_population(pos, new_pos, evals, new_evals, crossings, R, ru01);
  }
  cudaDeviceSynchronize();
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  cudaDeviceSynchronize();
  printf("\n");
  printf("[-] About your GPU ...\n");
  int dev = 0;
  cudaSetDevice(dev);

  cudaDeviceProp devProps;
  if (cudaGetDeviceProperties(&devProps, dev) == 0)
  {
      printf("[-]    Model            : %s \n[-]    Mem. global      : %d Bytes (%d MB) \n[-]    Compute cap.     : v%d.%d \n[-]    Clock speed      : %d MHz\n",
             devProps.name, (int)devProps.totalGlobalMem,
             (int)devProps.totalGlobalMem / 1000000,
             (int)devProps.major, (int)devProps.minor,
             (int)devProps.clockRate/1000);
  }

  printf("[-] About the experiment ...\n");
  printf("[-]    Function         : %s\n", F_NAME);
  printf("[-]    Heuristic method : DE\n");
  printf("[-]    Variables        : %d\n", N);
  printf("[-]    Population       : %d\n", POPULATION);
  printf("[-]    Bounds           : [%d, %d]\n", LB, UB);
  printf("[-]    Iters            : %d\n", N_ITERS);
  printf("[-] About the results ...\n");
  printf("[-]    Time required    : %f\n", cpu_time_used);
  printf("[-]    Minimum location : ");


  arrayReduction<<<1, POPULATION>>>(evals);
  print_iMin<<<1,1>>>(pos);

  cudaDeviceSynchronize();
  printf("\n");
  printf("\n");
  return 0;
}
