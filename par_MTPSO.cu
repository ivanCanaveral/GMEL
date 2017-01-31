#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>

#define WARP_SIZE 32

#define N 64
#define GRAPH "graphs/3Graph_64_0.txt"

#define N_PARTS 128
#define N_COL 3
#define N_ITERS 100
#define N_EXPERIMENTS 20

#define W 1
#define C1 2
#define C2 2
#define Vs 1e-05

__device__ int iMin = 0;
__device__ int fit_gm = 999999;

__global__ void show_global_fitness(){
  printf("%d\n", fit_gm);
}

// Atomic minimum operation
__device__ float atomicMinIndex(int * array, int * address, int val){
  int lo_que_tengo, lo_que_tenia;
  lo_que_tengo = * address;
  while (array[val] < array[lo_que_tengo]){
    lo_que_tenia = lo_que_tengo;
    lo_que_tengo = atomicCAS(address, lo_que_tenia, val);
  }
  return lo_que_tengo;
}

// Atomic max index operation
__device__ float atomicMaxIndex(int * array, int * address, int val){
  int lo_que_tengo, lo_que_tenia;
  lo_que_tengo = * address;
  while (array[val] > array[lo_que_tengo]){
    lo_que_tenia = lo_que_tengo;
    lo_que_tengo = atomicCAS(address, lo_que_tenia, val);
  }
  return lo_que_tengo;
}


// Inits all particles
__global__ void init_pos(int * pos, curandState_t * states){
  /* Lanzamos N_PARTS bloques de N hilos */
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  pos[id] = curand(&states[id]) % N_COL;
}

// Inits speed
__global__ void init_vel(float * vel, curandState_t * states){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  vel[id] = (curand_uniform(&states[id]) - 0.5) * 2 * N_COL;
}

// Fills an array with zeros (int)
__global__ void zeros(int * arr){
  /* As many threads and blocks as needed */
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  arr[id] = 0;
}

// Fills an array with a infty. (very high) value in all positions
__global__ void infty(int * arr){
    /* As many threads and blocks as needed */
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  arr[id] = 999999;
}
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

// // Prints an array of integers from Device
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

// Prints conflict matrix
void printCo(int * co){
  int i;
  for (i = 0; i < N_PARTS; i ++){
    icudaPrint(&co[i * N*N], N*N, N);
    printf("\n");
  }
}


__device__ float Cf(int c){
  return 1/(1 + exp((float)(-c + 2)));
}

__device__ float S(float x){
  return (1/(1 + exp(-x)));
}

__device__ int M(float Vj, float Cfj, float r){
  float output;
  output = 0;
  if ((Cfj > r) && (S(Vj) > r)){
    output = 1;
  }
  return output;
}

// Inits random states
__global__ void init_states(unsigned int seed, curandState_t * states) {
  /* we have to initialize the state */
  curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}

// Llena un array de randoms
__global__ void init_rands(float * arr, curandState_t * states){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  arr[id] = curand_uniform(&states[id]);
}

// Computes all conflict matrix for all particles
__global__ void compute_cos(int * Pos, int * A, int * co){
  int part = blockIdx.y;
  int i = blockIdx.x;
  int j = threadIdx.x;
  int * coloring;

  coloring = &Pos[part * N];

  if ((i != j) && (coloring[i] == coloring[j])){
    co[part * (N*N) + i * N + j] = A[i * N + j];
  } else {
    co[part * (N*N) + i * N + j] = 0;
  }
}

// Computes Cr for all particles
__global__ void compute_Cr(int * co, int * cr){
  /* N x N_PARTS blocks of N threads needed */
  int part = blockIdx.y;
  int i = blockIdx.x;
  int j = threadIdx.x;
  if (j == 0){
    cr[part * N + i] = 0;
  }
  __syncthreads();
  atomicAdd(&cr[part * N + i], co[part * N * N + i * N + j]);
}

__global__ void compute_Cr2(int * co, int * cr){
  /* N x N_PARTS blocks of N threads needed */
  int i = blockIdx.x;
  int j = threadIdx.x;
  int value = co[i * N + j];
  // We'll have Several warps Within a block, so we Have to synchronize
  if(j == 0){cr[i] = 0;}
  // Reduce step
  int offset;
  //  ()>> 1) is faster than (/2)
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value += __shfl_down(value, offset);
  }
  // X & (WARP_SIZE - 1) is faster than (X % WARP_SIZE)
  if ((j & 31) == 0) {atomicAdd(&cr[i],value);}
}

// Computes Cr
__global__ void compute_fitness(int * cr, int * fit){
  /* N_PARTS blocks with N threads */
  int part = blockIdx.x;
  int i = threadIdx.x;

  if (i == 0){
    fit[part] = 0;
  }
  __syncthreads();
  atomicAdd(&fit[part], cr[part * N + i]);
}

__global__ void compute_fitness_from_Cr(int * cr, int * fit){ // Yeah
  /* N * N_PARTS blocks of N threads */
  int i = blockIdx.x;
  int j = threadIdx.x;
  int value = cr[i * N + j];
  // We'll have Several warps Within a block, so we Have to synchronize
  if(j == 0){fit[i] = 0;}
  // Reduce step
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value += __shfl_down(value, offset);
  }
  if ((j & 31) == 0) {atomicAdd(&fit[i],value);}
}


__global__ void compute_fitness_from_Co(int * co, int * fit){
  /* Lanzamos N_PARTS x N bloques de N hilos */
  int i = blockIdx.y;
  int j = blockIdx.x;
  int k = threadIdx.x;
  int id = i * N * N + j * N + k;
  //printf("%d\n", id);
  int value = co[id];
  //printf("%d\n", value);
  if(j == 0){fit[i] = 0;}
  // Reduce step
  int offset;
  // Usamos >> 1 en lugar de /2
  /* De manera análoga a los ejemplos anteriores, vamos a sumar la mitad derecha
    del array con la mitad izquierda, pero edntro de cada warp */
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value += __shfl_down(value, offset);
  }
  if ((k & 31) == 0) {atomicAdd(&fit[i],value);}
}


// Calcula un array de longitud N_PARTS con los índices de los nodos más conflictivos de cada partícula
// Hay que mejorarlo haciendo reduce
__global__ void busca_max_conf_node(int * cr, int * max_confl_node){
  // Lanzamos 1 bloque, N_PARTS hilos
  // en cada particula ( su posicion es la coloración) buscamos el nodo con más conflictos
  // y lo ponemos en el array de max_confl_node
  int id = threadIdx.x;
  int i;
  int max = 0;
  for (i = 0; i < N; i++){
    if (cr[id * N + i] > cr[id * N + max]){
      max = i;
    }
  }
  max_confl_node[id] = max;
}

// Calcula un array de longitud N_PARTS con los índices de los nodos más conflictivos de cada partícula
// cr es de tamaño N*N_PARTS, y queremos reducirlo a un array de tamaño N
__global__ void busca_max_conf_node2(int * cr, int * max_confl_node){
  // Lanzamos N_PARTS bloque, N hilos
  // en cada particula ( su posicion es la coloración) buscamos el nodo con más conflictos
  // y lo ponemos en el array de max_confl_node
  int id = threadIdx.x;
  int thisThreadId = id;
  int value = cr[blockDim.x * blockIdx.x + id];
  //printf("m%d ",value );
  int gap, id2;
  int value2;
  for (gap = WARP_SIZE/2; gap > 0; gap >>= 1){
    id2 = __shfl_down(id, gap);
    value2 = __shfl_down(value, gap);
    //if (threadIdx.x == 0) printf("Gap : %d, value : %d, value2 %d, id : %d, id2 : %d\n", gap, value, value2, id2);
    if (value2 > value){
      value = value2;
      id = id2;
    }
  }
  if (((thisThreadId & (WARP_SIZE - 1)) == 0)){
    //printf("Intentamos colocar %d\n", id % N);
    atomicMaxIndex(&cr[blockIdx.x * blockDim.x], &max_confl_node[blockIdx.x], id);
  }
}


//Crea una nueva matriz de posiciones, cambiando las posiciones de máximo conflicto por el i indicado
// Nos servirá para buscar los mejores cambios de las posiciones conflictivas
__global__ void new_pos(int * pos, int * max_confl_node, int icolor, int * out){
  /* LAnzamos n_parts bloques y n hilos */
  int i = blockIdx.x;
  int j = threadIdx.x;
  if (max_confl_node[i] == j){
    out[i * N + j] = icolor;
  } else {
    out[i * N + j] = pos[i * N + j];
  }
}


// Creamos un array formado por los fitness de cada partícula para cada uno de los cambios
// tendrá longitud 4 x N_PARTS. Después usaremos este array para buscar el mejorarlo
// cambio en cada partícula
void crea_fitness_comp_array(int * pos, int * max_confl_node, int * a, int * comp_fit){
  // En esta función creamos el array de los fitness de los diferentes
  // cambios realizados en los nodos de mayor conflicto

  int * niu_pos, * co, * cr;
  cudaMalloc((void **) &niu_pos, N * N_PARTS * sizeof(int));
  cudaMalloc((void **) &co, N * N * N_PARTS * sizeof(int));
  cudaMalloc((void **) &cr, N * N_PARTS * sizeof(int));

  int color;
  for (color = 0; color < N_COL; color ++){
    // Creamos la nueva matriz de posiciones
    new_pos<<<N_PARTS, N>>>(pos, max_confl_node, color, niu_pos);

    // Creamos las matrices de conflictos, co, en niu_pos
    dim3 gridSize = dim3 (N, N_PARTS);
    dim3 blockSize = dim3 (N, 1);
    compute_cos<<<gridSize, blockSize>>>(niu_pos, a, co);

    // Computamos el fitness de cada una
    compute_Cr<<<gridSize, blockSize>>>(co, cr);

    compute_fitness<<<N_PARTS, N>>>(cr, &comp_fit[color * N_PARTS]);
  }
  cudaFree(niu_pos);
  niu_pos = NULL;
}

void crea_fitness_comp_array2(int * pos, int * max_confl_node, int * a, int * comp_fit, int * co){
  // En esta función creamos el array de los fitness de los diferentes
  // cambios realizados en los nodos de mayor conflicto

  int * niu_pos;
  cudaMalloc((void **) &niu_pos, N * N_PARTS * sizeof(int));
  dim3 gridSize = dim3 (N, N_PARTS);
  dim3 blockSize = dim3 (N, 1);

  int color;
  for (color = 0; color < N_COL; color ++){
    // Creamos la nueva matriz de posiciones
    new_pos<<<N_PARTS, N>>>(pos, max_confl_node, color, niu_pos);
    // Creamos las matrices de conflictos, co, en niu_pos
    compute_cos<<<gridSize, blockSize>>>(niu_pos, a, co);
    compute_fitness_from_Co<<<gridSize, blockSize>>>(co, &comp_fit[color * N_PARTS]);
    // Computamos el fitness de cada una
    //compute_Cr<<<gridSize, blockSize>>>(co, cr);
  }
  cudaFree(niu_pos);
  niu_pos = NULL;
}

// Buscámos los mínimos de los fitness cambiados. Además los comparamos con
// los de las coloraciones (pos) actualies. Si no mejoran la solución ponemos
// un -1 en su lugar.
__global__ void find_fitness_mins(int * fit_comp, int * fit_mins, int * fit){
  /* Lanzamos N_PARTS hilos */
  int id = threadIdx.x;
  int min = 0;
  int i;
  // Asumimos este bucle, ya que sólo tiene 4 iteraciones
  for (i = 0; i < N_COL; i++){
    if (fit_comp[i * N_PARTS + id] < fit_comp[min * N_PARTS + id]) {
      min = i;
    }
  }
  // Si no mejoramos la solución que había, ponemos -1
  if (fit_comp[min * N_PARTS + id] < fit[id]){
    fit_mins[id] = min;
  } else {
    fit_mins[id] = -1;
  }
}

// Mueve una partícula siguiendo el esquema paso
// Lo usaremos cuando no mejoremos al intentar resolver conflictos
__global__ void move(int * pos, int * cr, float * vel, float * r){
  /* lanzamos 1 bloque con N hilos */
  int j = threadIdx.x;

  float Cfj = Cf(cr[j]);
  float Vj = vel[j];
  int MVj = M(Vj, Cfj, r[j]);
  pos[j] = (pos[j] + MVj) % 4;
}

// Cambia la posición. Si se mejora resolviendo nodos conflictivos, lo hace. En
// Caso contrario, sigue el movimiento usando MTPSO
// fit_mins contiene los mejores cambios, y en caso de no existir, habrá -1
__global__ void update_pos(int * pos, int * fit_mins, int * max_confl_node, int * cr, float * vel, float * r){
  /* Lanzamos N_PARTS hilos*/
  int id = threadIdx.x;
  // Si tenemos un -1 movemos la partícula
  if (fit_mins[id] < 0){
    move<<<1,N>>>(&pos[id * N], &cr[id * N], &vel[id * N], &r[id * N]);
    cudaDeviceSynchronize();
  } else {
    // En caso contrario cambiamos el nodo conflictivo
    int place = max_confl_node[id];
    pos[id * N + place] = fit_mins[id];
  }
}

// Dadas las posiciones y la matriz a, completa el vector de fitness (out)
// Esto se puede hacer directamente en un kernel con reduce. Cambiar!
void compute_fitness(int * pos, int * a, int * out){
  int * co, * cr;
  cudaMalloc((void **) &co,  N * N * N_PARTS * sizeof(int));
  cudaMalloc((void **) &cr, N * N_PARTS * sizeof(int));

  dim3 gridSize = dim3 (N, N_PARTS);
  dim3 blockSize = dim3 (N, 1);
  compute_cos<<<gridSize, blockSize>>>(pos, a, co);
  compute_Cr<<<gridSize, blockSize>>>(co, cr);
  compute_fitness<<<N_PARTS, N>>>(cr, out);

  cudaFree(co);
  cudaFree(cr);
}


// Copia tantos elementos como hilos de el array1 al array2
__global__ void copy(int * arr1, int * arr2){
  /* Lanzamos tantos hilos como elementos queramos copiar */
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  arr2[i] = arr1[i];
}

__global__ void update_min(int * pos, int * min, int * fit_pos, int * fit_min){
  /* Lanzamos N_PARTS hilos */
  int i = threadIdx.x;
  if (fit_pos[i] < fit_min[i]){
    copy<<<1, N>>>(&pos[i*N], &min[i*N]);
  }
}

__global__ void update_min2(int * pos, int * min, int * fit_pos, int * fit_min){
  /* Lanzamos N_PARTS hilos */
  int i = threadIdx.x;
  if (fit_pos[i] < fit_min[i]){
    copy<<<1, N>>>(&pos[i*N], &min[i*N]);
    fit_min[i] = fit_pos[i];
  }
}

// Calcula el íncice del elemento mínimo en un array, y lo guarda en iMin
__global__  void arrayReduction(int * array){
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int thisThreadId = id;
  int value = array[id];
  int gap, id2;
  int value2;
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

// Cambia el valor de gMin si procede, y también de fit_gm
__global__ void change_gMin(int * gMin, int * min, int * fit_m){
  /* lanzamos N hilos */
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int fitness_gMin = fit_gm;
  __syncthreads();
  if (fit_m[iMin] < fitness_gMin){
    gMin[id] = min[iMin * N + id];
    if (id == 0){fit_gm = fit_m[iMin];}
  }
}

__global__ void minMin(int * fit_min){
  int i;
  int i_Min = 0;
  for (i = 0; i < N_PARTS; i++){
    if (fit_min[i] < fit_min[i_Min]){
      i_Min = i;
    }
  }
  iMin = i_Min;
}

__global__ void compute_fit_gm(int * gco){
  /* LAnzamos N bloques de N hilos*/
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id == 0){ fit_gm = 0;}
  __syncthreads();
  atomicAdd(&fit_gm, gco[id]);
}

__global__ void gMin_CAS(int * gMin, int * min, int * fit_min){
  /* Lanzamos N hilos */
  int id = threadIdx.x;
  if (fit_min[iMin] < fit_gm){
    gMin[id] = min[iMin * N + id];
  }
}

// Actualiza el mínimo global
void update_gMin(int * gmin, int * min, int * fit_m, int * a){
  // Creamos la variable que se encargará de albergar la matriz de conflictos de gMin
  int * gCo;
  cudaMalloc((void **) &gCo, N * N * sizeof(int));
  // Calculamos la matriz de conflictos
  compute_cos<<<N,N>>>(gmin, a, gCo);

  // Calculamos el fitness del minimo global (se guarda en la variable device fit_gm)
  compute_fit_gm<<<N,N>>>(gCo);

  // Calculamos el índice del mínimo entre los mínimos
  minMin<<<1,1>>>(fit_m);

  // Actualizamos gMin si procede
  gMin_CAS<<<1,N>>>(gmin, min, fit_m);
}

__global__ void update_vel(float * vel, int * pos, int * min, int * gMin, float * r1, float * r2){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int i = threadIdx.x;
  vel[id] = vel[id] * W  + C1 * r1[id] * (min[id] - pos[id]) + C2 * r2[id] * (gMin[i] - pos[id]);
}


// Imprime un array de floats
void fMatrixPrint(float * array, int elements, int n_jump){
  int i;
  for (i = 0; i < elements; i++){
      if ((i % n_jump) == 0 ) {printf("\n");}
      printf("%f ", array[i]);
  }
}

// Imprime un array de ints
void iMatrixPrint(int * array, int elements, int n_jump){
  int i;
  for (i = 0; i < elements; i++){
      if ((i % n_jump) == 0 ) {printf("\n");}
      printf("%d ", array[i]);
  }
}

__global__ void reset(){
  fit_gm = 999999;
}


// Min from arr
int immin(int * arr, int n){
  int mmin = arr[0];
  int i;
  for (i = 1; i < n; i ++){
    if (arr[i] < mmin){
      mmin = arr[i];
    }
  }
  return mmin;
}

// Return the integer mean of the n first elements from arr
int imean(int * arr, int n){
  int ssum = 0;
  int i;
  for (i = 0; i < n; i++){
    ssum += arr[i];
  }
  return ssum/n;
}


// Std deviation from the n first elements of arr
float idev(int * arr, int mean, int n){
  float ssum = 0;
  int i;
  for (i = 0; i < n; i++){
    ssum += pow(arr[i] - mean, 2);
  }
  return sqrt(1/(float)(n - 1) * ssum);
}


// Returns the number of zeros in an array of size N
int count_zeros(int * arr, int n){
  int i;
  int count;
  for (i = 0; i < n; i++){
    if (arr[i] == 0){
      count += 1;
    }
  }
  return count;
}

int main(void){

  int ady[N*N];

  // Loading from file...
  FILE * my_file;
  my_file = fopen(GRAPH,"r");
  int k;
  for (k = 0; k < N*N; k++){
    //printf("%d\n", k);
    fscanf(my_file, "%d", &ady[k]);

  }

  fclose(my_file);

  int * A;

  // Copying to GPU
  cudaMalloc((void **) &A, N * N * sizeof(int));
  cudaMemcpy(A, ady, N * N * sizeof(int), cudaMemcpyHostToDevice);

  int * Pos, * co, * cr, * fit, * max_confl_node;
  int * comp_fit, * fit_mins, * Min, * fit_m, * gMin;
  //int * changed;
  float * Vel;

  cudaMalloc((void **) &A, N * N * sizeof(int));
  cudaMalloc((void **) &Pos, N * N_PARTS * sizeof(int));
  cudaMalloc((void **) &co, N * N * N_PARTS * sizeof(int));
  cudaMalloc((void **) &cr, N * N_PARTS * sizeof(int));
  cudaMalloc((void **) &fit, N_PARTS * sizeof(int));
  cudaMalloc((void **) &max_confl_node, N_PARTS * sizeof(int));
  cudaMalloc((void **) &comp_fit, N_COL * N_PARTS * sizeof(int));
  cudaMalloc((void **) &fit_mins, N_PARTS * sizeof(int));
  cudaMalloc((void **) &fit_mins, N_PARTS * sizeof(int));
  cudaMalloc((void **) &Vel, N * N_PARTS * sizeof(float));
  cudaMalloc((void **) &Min, N * N_PARTS * sizeof(int));
  cudaMalloc((void **) &fit_m, N_PARTS * sizeof(int));
  cudaMalloc((void **) &gMin, N * sizeof(int));

  // Generamos los estados
  curandState_t * states;
  cudaMalloc((void**) &states, N * N_PARTS * sizeof(curandState_t));
  init_states<<<N_PARTS*N,1>>>(time(0), states);

  // Generamos los números aleatorios
  float * r0, * r1, * r2;
  cudaMalloc((void **) &r0, N * N_PARTS * sizeof(float));
  cudaMalloc((void **) &r1, N * N_PARTS * sizeof(float));
  cudaMalloc((void **) &r2, N * N_PARTS * sizeof(float));

  init_pos<<<N_PARTS, N>>>(Pos, states);
  init_vel<<<N_PARTS, N>>>(Vel, states);
  zeros<<<N_PARTS,N>>>(Min);
  zeros<<<1,N>>>(gMin);
  infty<<<1,N_PARTS>>>(fit_m);

  dim3 gridSize = dim3 (N, N_PARTS);
  dim3 blockSize = dim3 (N, 1);

  clock_t start, end;

  int experiment;
  int solutions[N_EXPERIMENTS];

  cudaDeviceSynchronize();

  float total_time = 0;

  for (experiment = 0; experiment < N_EXPERIMENTS; experiment ++){
    start = clock();

    int iter;
    for (iter = 0; iter < N_ITERS; iter ++){

      compute_cos<<<gridSize, blockSize>>>(Pos, A, co);

      compute_Cr2<<<N * N_PARTS, N>>>(co, cr);

      compute_fitness_from_Cr<<<N_PARTS, N>>>(cr, fit);

      init_rands<<<N,N_PARTS>>>(r0, states);
      init_rands<<<N,N_PARTS>>>(r1, states);
      init_rands<<<N,N_PARTS>>>(r2, states);

      //    Actualizamos los mínimos
      //        Primero necesitamos tener los fitness previos (en este caso mu malos)
      update_min2<<<1, N_PARTS>>>(Pos, Min, fit, fit_m);

      //    Actualizamos el mínimo global
      //        Buscamos el menor de todos los mínimos, y guardamos su índice en iMin
      arrayReduction<<<1,N_PARTS>>>(fit_m);


      //        Cambiamos el valor de gMin si procede, y también fit_gm
      change_gMin<<<1,N>>>(gMin, Min, fit_m);
      //printf("\nesto es lo que hay en la posición iMin de Pos : \n");

      update_vel<<<N_PARTS,N>>>(Vel, Pos, Min, gMin, r1, r2);
      //fcudaPrint(Vel, N*N_PARTS, N);

      //    Actualizamos la posición
      //        En primer lugar, dado Cr, buscamos el nódo más conflictivo de cada partícula
      zeros<<<1,N_PARTS>>>(max_confl_node);
      busca_max_conf_node2<<<N_PARTS,N>>>(cr, max_confl_node);
      //        Creamos el array de la comparación de fitness
      // (!) Ojo, como ahora computamos todas las posibilidades, podemos buscar el mejor cambio
      // mejorando así la calidad de la solución y la velocidad de convergencia.
      // Este proceso se puede mejorar, creando cada fitness a partir de su co correspondiente mediante una única función
      crea_fitness_comp_array2(Pos, max_confl_node, A, comp_fit, co);
      //        En el array comp_fit buscamos el mejor cambio en cada uno, y lo guardamos en fit_mins
      find_fitness_mins<<<1,N_PARTS>>>(comp_fit, fit_mins, fit);
      //        Finalmente, actualizamos la posición
      update_pos<<<1,N>>>(Pos, fit_mins, max_confl_node, cr, Vel, r0);
    }

    end = clock();
    total_time += ((float) (end - start)) / CLOCKS_PER_SEC;
    cudaDeviceSynchronize();
    cudaMemcpy(&solutions[experiment], &fit_gm, sizeof(int), cudaMemcpyDeviceToHost);
    reset<<<1,1>>>();
    cudaDeviceSynchronize();
  }

  printf("\n--- MTPSO : Test info ---\n");
  printf("[-] Population       : %d\n", N_PARTS);
  printf("[-] Variables        : %d\n", N);
  printf("[-] Num. max iters   : %d\n", N_ITERS);
  printf("[-] Num. experiments : %d\n", N_EXPERIMENTS);
  printf("[-] Required time        : %f\n", total_time);
  printf("[-] Required time mean   : %f\n", total_time/N_EXPERIMENTS);

  int sol_mean = imean(solutions, N_EXPERIMENTS);

  printf("\n--- Results ---\n");
  printf("[-] Best solution : %d\n", immin(solutions, N_EXPERIMENTS));
  printf("[-]  Solution info :\n");
  printf("      Mean : %d\n", sol_mean);
  printf("      Dev  : %f\n", idev(solutions, imean(solutions, N_EXPERIMENTS), N_EXPERIMENTS));
  printf("[-] Success rate : %.2f \n", (float)count_zeros(solutions, N_EXPERIMENTS)/ N_EXPERIMENTS);


  cudaDeviceSynchronize();
  return 0;
}
