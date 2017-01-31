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

#define N 32   // Dimension
#define POPULATION 128 // Population
#define UB 10 // Upper bound
#define LB -10 // Lower bound

#define CR 0.8
#define F 0.5

#define WARP_SIZE 32
#define N_WARPS N/WARP_SIZE
#define N_TEAMS POPULATION/WARP_SIZE
#define N_ITERS 1000

//#define FUNCTION 10


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
  arr[id] = curand_normal(&states[id])/1;
}

// Computes random integers between 0 and n
__global__ void irand(int n, int * arr, curandState_t * states){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  arr[id] = curand(&states[id]) % n;
}

// Set up the initial population
__global__ void init_bees(float * pos, float * rand_uniform){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  pos[id] = LB + rand_uniform[id] * (UB - LB);
}

// Sets the counter array to zero
__global__ void init_counter(int * counter){
  // 1 block of N threads will be enough
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  counter[id] = 0;
}

// Fill an array with zeros
__global__ void zeros(int * arr){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  arr[id] = 0;
}

// Given a value, returns its fitness
__device__ float fitness(float value){
  float fit;
  if (value >= 0){
    fit = __fdiv_rn(1,__fadd_rn(1,value));
  } else {
    fit = __fadd_rn(1, fabsf(value));
  }
  return fit;
}

// Function to minimize : Sphere
__global__ void SPH(float * x, float * evals){
  // Launch 1 block of N threads for each element fo the population: <<<POPULATION, N>>>

  // variable sum is shared by all threads in each block
  __shared__ float sum;
  //__shared__ float prod;
  //__shared__ float sum2;

  float value;
  int i = blockIdx.x * blockDim.x + threadIdx.x;


  // Init the sum value to 0
  if (threadIdx.x == 0) {sum = 0;}
  // The main operation
  //value = __fmul_rd(__fsub_rd(x[i],1),__fsub_rd(x[i],1));
  value = pow(x[i],2);

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
      evals[blockIdx.x] = fitness(sum);
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
  // The main operation (fast version)
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

  // Only one thread writes the result of this block-bee
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

  // The main operation (fast version)
  //value = __fmul_rd(__fsub_rd(x[i],1),__fsub_rd(x[i],1));
  value0 = pow(x[i],2);
  value1 = 0.5 * threadIdx.x * (x[i]);
  //printf("tdt %f\n", value0);

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
      //printf("%f\n", sum);
      //value = sum / 10000;
      evals[blockIdx.x] = sum; //<<<<<<<<<<<<<<<<<<<<<<<<<----------!!! Esa puta mierda tendúa que ser sum, pero si lo pongo peta
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

  // The main operation (fast version)
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
  } else if (threadIdx.x == 1){
    addend1 = __expf(__fmul_rd(__frcp_rz(N), value1));
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

  // The main operation (fast version)
  //value = __fmul_rd(__fsub_rd(x[i],1),__fsub_rd(x[i],1));
  value0 = pow(x[i], 2);
  value1 = __cosf(__fdiv_rn(x[i], __fsqrt_rn(threadIdx.x + 1)));

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

  // The main operation (fast version)
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

// Move a bee to a new source in the neighbourhood
__global__ void seek_around(float * bees, float * new_sources, float * rand_uniform, int * crossings){
  // As many blocks as bees, as many threads per blocks as dimensio
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int k = crossings[blockIdx.x]; // the same crossing for the whole block
  float phi = rand_uniform[id]; // A new phi for each thread
  new_sources[id] = bees[id] + abs(phi) * (bees[k * N + threadIdx.x] - bees[id]);
}

// Keeps the best source_info (with counter)
__global__ void keep_best_sources_c(float * sources, float * new_sources, float * fit, float * new_fit, int * counter){
  // It is necessary to launch POPULATION blocks and N threads
  //    in this case we have to assume some thread divergence
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.x;

  // Compare new and old solution
  if (new_fit[i] > fit[i]){
    // Update the fitness
    if (threadIdx.x == 0){fit[i] = new_fit[i];}
    // Copy the new position
    sources[id] = new_sources[id];
    // Reset the counter
    if (threadIdx.x == 0){counter[i] = 0;}
    //if (threadIdx.x == 0){printf("yes\n");}
  } else {
    if (threadIdx.x == 0){counter[i] += 1;}
    //if (threadIdx.x == 0){printf("nope\n");}
  }
}


// Resume warp probs (We assume that the number of bees will be less than 1024)
__global__ void warp_reduce(float * fitness, float * warp_result){
  // We use 1 block of N_BEES threads. This will limit the number of bees to 1024.
  // If we want to use more bees, we can launch (N_BEES/1024 + 1) blocks
  int i = threadIdx.x; // Thread id
  //int j = blockIdx.x; // Block id

  // Get the value
  float value = fitness[i];
  //printf("%f\n", value);
  // Do the reduce
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value += __shfl_down(value, offset);
  }
  // Only the first thread of each block writes the result
  if ((i & 31) == 0){
    int idx = i/WARP_SIZE;
    //printf("%f guardado en %d\n", value, idx);
    warp_result[idx] = value;
  }
}

__global__ void roulette1(float * in, int n, int * out, float * rands){
  /* M bloques de WARP_SIZE elementos */
  __shared__ int auxiliar;
  auxiliar = 0;
  int id = threadIdx.x;
  float value = in[id];
  float val;
  #pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1){
    val = __shfl_up(value, offset);
    if ((id % 32) >= offset) {
      value += val;
    }
  }
  // Rescatamos el valor del último elemento (n-1)
  float max = __shfl(value, n-1);
  value = value/max;
  // Get a random value
  float r;

  r = rands[blockIdx.x];

  bool brick = r > value;

  if(threadIdx.x < n){atomicAdd(&auxiliar, (int)brick);}

  if (threadIdx.x == 0){
    out[blockIdx.x] = auxiliar;
  }
}

__global__ void roulette2(float * probs, int * leaders, int * out, float * rands){
  /* N es el array con los datos de entradoa y N
  el tamaño de dicho array */
  __shared__ int auxiliar;
  auxiliar = 0;
  //int i = threadIdx.x;
  int id = leaders[blockIdx.x] * 32 + threadIdx.x;
  float value = probs[id];
  float val;
  #pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1){
    val = __shfl_up(value, offset);
    if ((id % 32) >= offset) {
      value += val;
    }
  }
  // Rescatamos el valor del último elemento (n-1)
  float max = __shfl(value, 31);
  value = value/max;
  // Conseguimos un valor aleatorio
  float r;
  r = rands[blockIdx.x];
  bool brick = r > value;
  atomicAdd(&auxiliar, (int)brick);
  if ((id & 31) == 0){
    out[blockIdx.x] = auxiliar + leaders[blockIdx.x] * 32;
  }
}

// Sends the onlookers to their destinations
__global__ void send_onlookers(float * onlookers, float * employees, int * destinations, float * rand_normal, int * crossings){
  // Each block will set one onleeker, so we'll need one block of N threads for each onlooker
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  float r = rand_normal[POPULATION * N - id - 1]; // Take it in reverse order, avoiding generate more rands
  int k = crossings[POPULATION - blockIdx.x - 1]; // Reverse too
  int dest = destinations[blockIdx.x];
  int j = threadIdx.x;

  //onlookers[id] = employees[dest * N + j] + r * (employees[dest * N + j] - employees[k * N + j]);
  onlookers[id] = employees[dest * N + j] + abs(r)/3 * (employees[k * N + j] - employees[dest * N + j]);
}


// Operación atómica que escribe el mínimo de un índice
__device__ float atomicMaxIndex(int * array, int * address, int val){
  int lo_que_tengo, lo_que_tenia;
  lo_que_tengo = * address;
  printf("(%d, %d), Esto : %f debe ser menor que esto %f\n", val, lo_que_tengo, array[val], array[lo_que_tengo]);
  while (array[val] > array[lo_que_tengo]){
    lo_que_tenia = lo_que_tengo;
    lo_que_tengo = atomicCAS(address, lo_que_tenia, val);
  }
  return lo_que_tengo;
}

// Fills the array of changes
//    in each block it looks for the best onlooker and save it in changes array
__global__ void set_changes(float * ofit, int * destinations, int * changes){
  // 1 block per bee, and POPULATION threads per block : <<<POPULATION, POPULATION>>>

  //__shared__ int best_onlooker;
  __shared__ int winners_onlokers[N_TEAMS];
  __shared__ float winners_values[N_TEAMS];

  //if (threadIdx.x == 0) {best_onlooker = 0;}

  int employee = blockIdx.x;
  int onlooker = threadIdx.x;
  float value, value2;

  // Set the values
  if (employee == destinations[onlooker]){
    value = ofit[onlooker];
  } else {
    value = -1;
  }

  // Write the value in the shared array
  //values[onlooker] = value;
  __syncthreads();

  // Do the reduction with shuffle in order to get a big speed up
  int gap, onlooker2;
  for (gap = WARP_SIZE/2; gap > 0; gap >>= 1){
    onlooker2 = __shfl_down(onlooker, gap);
    value2 = __shfl_down(value, gap);
    if (value2 > value){
      value = value2;
      onlooker = onlooker2;
    }
  }

  // Save the warp winners in a shared memory array in order to compare it
  if (((threadIdx.x & (WARP_SIZE - 1)) == 0)){
    int i = (int) threadIdx.x / WARP_SIZE;
    winners_onlokers[i] = onlooker;
    winners_values[i] = value;
  }

  __syncthreads();
  // Compare it
    if (threadIdx.x < N_TEAMS){
      value = winners_values[threadIdx.x];
      onlooker = winners_onlokers[threadIdx.x];
      for (gap = N_TEAMS/2; gap > 0; gap >>= 1){
        onlooker2 = __shfl_down(onlooker, gap);
        value2 = __shfl_down(value, gap);
        if (value2 > value){
          value = value2;
          onlooker = onlooker2;
        }
      }

    __syncthreads();
    if (threadIdx.x == 0){
      if (value < 0){
        changes[employee] = -1;
      } else {
        changes[employee] = onlooker;
      }
    }
  }
}

// Update the employess with the info obteained by onlookers
__global__ void make_changes(float * employees, float * onlookers, float * ofit, float * efit, int * changes){
  // One block per employee, with N threads
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int employee = blockIdx.x;

  // Check if the change we'll carry the employee to a better place
  if (changes[employee] > -1){
    //printf("Intenta cambiar : %f - %f \n", efit[employee], ofit[changes[employee]] );
    if (efit[employee] < ofit[changes[employee]]){
      // If the new source is better, change employee's mind
      //printf("CAmbio\n");
      employees[id] = onlookers[changes[employee] * N + threadIdx.x];
    }
  }
}


// Sets up test population
__global__ void test_bees(float * employees){
  // 1 block per employye with n threads
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  employees[id] = (blockIdx.x + 3)/2.0;
}


__global__ void test(float * arr){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  arr[id] = 1;
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
//  a member is only replaced by a better solution, so we can only use this
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
    atomicMinIndex(array, &iMin, id);
  }
}

__global__ void print_iMin(float * pos){
  int i;
  for (i = 0; i < N; i++){
    printf("%f ", pos[iMin * N + i]);
  }
}

int main(void){

  // States
  curandState_t * states;
  cudaMalloc((void**) &states, N * POPULATION * sizeof(curandState_t));
  init_states<<<N*POPULATION,1>>>(time(0), states);

  // The random things
  float * ru01, *ru02, * rn01, * rn02;
  int * crossings;
  cudaMalloc((void **) &ru01, N * POPULATION *sizeof(float));
  cudaMalloc((void **) &ru02, N * POPULATION *sizeof(float));
  cudaMalloc((void **) &rn01, N * POPULATION *sizeof(float));
  cudaMalloc((void **) &rn02, N * POPULATION *sizeof(float));
  cudaMalloc((void **) &crossings, POPULATION * sizeof(int));

  U_01<<<N, POPULATION>>>(ru01, states);
  U_01<<<N, POPULATION>>>(ru02, states);
  N_01<<<N, POPULATION>>>(rn01, states);
  N_01<<<N, POPULATION>>>(rn02, states);

  irand<<<1,POPULATION>>>(POPULATION, crossings, states);

  // All the stuff
  float * employees, * onlookers, * destinations, * new_sources;
  float * efit, * ofit, * new_fit, * dest_fit;
  float * warp_sums;
  int * counter, * destination, * changes;

  cudaMalloc((void **) &employees, N * POPULATION * sizeof(float));
  cudaMalloc((void **) &onlookers, N * POPULATION * sizeof(float));
  cudaMalloc((void **) &destinations, N * POPULATION * sizeof(float));
  cudaMalloc((void **) &new_sources, N * POPULATION * sizeof(float));
  cudaMalloc((void **) &efit, POPULATION * sizeof(float));
  cudaMalloc((void **) &ofit, POPULATION * sizeof(float));
  cudaMalloc((void **) &new_fit, POPULATION * sizeof(float));
  cudaMalloc((void **) &dest_fit, POPULATION * sizeof(float));
  cudaMalloc((void **) &warp_sums, POPULATION/WARP_SIZE * sizeof(float));
  cudaMalloc((void **) &counter, POPULATION * sizeof(int));
  cudaMalloc((void **) &destination, POPULATION * sizeof(int));
  cudaMalloc((void **) &changes, POPULATION * sizeof(int));

  init_counter<<<1,POPULATION>>>(counter);
  init_bees<<<POPULATION, N>>>(employees, ru01);

  test_bees<<<POPULATION, N>>>(employees);
  cudaDeviceSynchronize();
  printf("\n---go---\n");
  clock_t start, end;
  double cpu_time_used;

  start = clock();

  //generate_random_things(states, ru01, R, crossings);
  int i;
  for (i = 0; i < N_ITERS; i ++){

    // Employees ----------------------------------
    // Compute the fitness of the employees
    F_KEY<<<POPULATION, N>>>(employees, efit);
    seek_around<<<POPULATION,N>>>(employees, new_sources, ru01, crossings);
    F_KEY<<<POPULATION, N>>>(new_sources, new_fit);
    // Keep the best solution
    keep_best_sources_c<<<POPULATION, N>>>(employees, new_sources, efit, new_fit, counter);

    // Onlookers ----------------------------------
    // set destination using roulette
    //  reduce the warps
    warp_reduce<<<1, POPULATION>>>(efit, warp_sums);
    //  select the destination warps
    roulette1<<<POPULATION, WARP_SIZE>>>(warp_sums, POPULATION/WARP_SIZE, destination, ru02); // Save the choosen warps
    roulette2<<<POPULATION, WARP_SIZE>>>(efit, destination, destination, ru01); // Save the choosen destination

    // Sending onlookers
    send_onlookers<<<POPULATION, N>>>(onlookers, employees, destination, rn01, crossings);

    // Evaluate new sources
    F_KEY<<<POPULATION, N>>>(onlookers, ofit);

    // Updating employees
    set_changes<<<POPULATION, POPULATION>>>(ofit, destination, changes);
    make_changes<<<POPULATION, N>>>(employees, onlookers, ofit, efit, changes);
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
    printf("[-]    Heuristic method : ABC\n");
    printf("[-]    Variables        : %d\n", N);
    printf("[-]    Population       : %d\n", POPULATION);
    printf("[-]    Bounds           : [%d, %d]\n", LB, UB);
    printf("[-]    Iters            : %d\n", N_ITERS);
    printf("[-] About the results ...\n");
    printf("[-]    Time required    : %f\n", cpu_time_used);
    printf("[-]    Minimum location : ");


    arrayReduction<<<1, POPULATION>>>(efit);
    print_iMin<<<1,1>>>(employees);

    cudaDeviceSynchronize();
    printf("\n");
    printf("\n");
    return 0;
}
