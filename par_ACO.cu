#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>

#define N_EXPERIMENTS 100

#define N 128
//#define MAP "maps/128Circ.txt"
#define MAP "maps/128bier.txt"
//#define N 256
//#define MAP "maps/256Circ.txt"
//#define N 384
//#define MAP "maps/384Circ.txt"
//#define N 512
//#define MAP "maps/512Circ.txt"
//#define N 640
//#define MAP "maps/640Circ.txt"


#define WARP_SIZE 32
#define N_WARPS N/WARP_SIZE
#define N_ITERS 200

#define ALPHA 1
#define BETA 2

#define RHO 0.5
#define C N/8


__device__ int iMin = 0;

// Imprime un array de floats en Device
void fcudaPrint(float * array, int elements, int n_jump){
  float * aux;
  aux = (float *) malloc(elements * sizeof(float));
  cudaMemcpy(aux, array, elements * sizeof(float), cudaMemcpyDeviceToHost);

  int i;
  for (i = 0; i < elements; i++){
      if ((i % n_jump) == 0 ) {printf("\n");}
      printf("%.1f ", aux[i]);
  }
  free(aux);
  aux = NULL;
}

// Imprime un array de ints en Device
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

__device__ int isInPath(int ciudad, int * recorrido, int n){
  /* comprueba si la ciudad está en el recorrido.
    n es la logitud del recorrido. puede ser la
    longitud actual o la total

    Ojo, si la ciudad no está en la lista devilvemos uno.
    Si está, 0. */
  int i;
  int output = 1;
  for (i = 0; i<n; i++){
    if (recorrido[i] == ciudad) {
      output = 0;
    }
  }
  return output;
}

__global__ void init_states(unsigned int seed, curandState_t * states) {
  /* we have to initialize the state */
  //curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
  curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}

__global__ void init_rands(float * arr, curandState_t * states){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  arr[id] = curand_uniform(&states[id]);
}

__global__ void init_A(float * a){
  // N bloques, N hilos
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int i = threadIdx.x;
  int j = blockIdx.x;
  if (i==j){
    a[id] = 999999;
  } else if ((__usad(i,j,0)==1) || (i == 0 && j == N-1) || (j == 0 && i == N-1)) {
    a[id] = 1;
  } else {
    a[id] = 1.2;
  }
}

__global__ void init_pos_actual(int * pos_actual){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  pos_actual[id] = id;
}

__global__ void init_K(int * k){
  // lanzamos un único hilo;
  *k = 0;
}

// Llena un array con un número x
__global__ void fill(float * arr, float x){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  arr[id] = x;
}

// LLena un array con un entero x
__global__ void ifill(int * arr, int x){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  arr[id] = x;
}

// Toma la matriz de adyacencia, la de las feromonas, los caminos y generaremos
// un array de NxN con las probabilidades de ir desde i hasta j
__global__ void set_probs(float * a, float * f, int * path, int * pos_actual, float * out){
  /* Asumimos que generaremos N bloques con N hilos */
  /*
    Cambiamos esta función, añadiendo un array de posición actual
    para que puedan representarse los cambios de posición de las hormigas.

    Esto afecta al la lectura de la matriz de adyacencia, a la de feromonas
    y también a los caminos. Para cambiar todo esto al mismo Tiempo
    cambiaremos cómo se calcula la id. cambiaremos :

            int id = blockIdx.x * blockDim.x + threadIdx.x;
    por :
        int id = pos_actual[blockIdx.x] * blockDim.x + threadIdx.x;

    y así la información de las probabilidades se cargará ya con la información
    de las ciudades de origen.
  */
  __shared__ int cities[N];
  // La ciudad de destino que maneja este hilo
  int i = threadIdx.x;
  int id = pos_actual[blockIdx.x] * blockDim.x + threadIdx.x;
  // Copiamos la ruta a la memoria compartida
  //   Cuidado, que estas ciudades hay que cogerlas de la fila "buena" de paths, no de la edsplazada
  cities[i] = path[blockIdx.x * blockDim.x + threadIdx.x];
  // Sincronizamos para que todos los hilos del bloque hayan copiado
  __syncthreads();
  // vemos si está en el camino. En este caso Id coincide con el número de ciudad
  int taboo;
  taboo = isInPath(i, cities, N);
  //printf("%d\n", taboo);
  // calculamos la probabilidad sin el denominador
  float prob;
  prob = __fmul_ru(__fdiv_ru(1,a[id]),f[id]);
  //printf("%d, %f\n", taboo, prob);
  // calculamos el valor definitivo de la probabilidad
  /*
  Ojito, que aquí no nos vale el id anterior, porque si no únicamente escribiríalojamos
  en una única fila. Ahora tenemos que calcular el id de verdad, para que lo escriba
  en el lugar correspondiente en la matriz de probabiliae
  */
  out[blockIdx.x * blockDim.x + threadIdx.x] = __fmul_ru(taboo, prob);
}

// Dado el array de las probabilidades, rellena Steve
__global__ void sum_up(float * probs, float * out){
  /* Asumimos que generaremos N bloques con N hilos */

  int i = threadIdx.x;
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  float value;
  value = probs[id];
  // ahora hacemos el paso de warp_reduction
  //  recordamos que out será el resultado de la div. entera N/WARP_SIZE
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value += __shfl_down(value, offset);
  }
  if ((id & 31) == 0){
    // el índice del array out será la división entera de id entre 32
    int idx = i/WARP_SIZE;
    out[blockIdx.x * WARP_SIZE + idx] = value;
  }
}


// Le damos Steve y llena el array de los warps elegidos
__global__ void roulette1(float * in, int n, int * out, float * rands){
  /* N es el array con los datos de entradoa y N
  el tamaño de dicho array */
  __shared__ int auxiliar;
  auxiliar = 0;
  int id = blockIdx.x * blockDim.x + threadIdx.x;
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
  //if (id < n) {in[id] = value;}

  // Conseguimos un valor aleatorio
  float r;
  /*if (id == 0){
    // Si es el hilo cero lo generamos
    r = curand_uniform(&states[id]);
  } else {
    // si no, se lo copiamos al hilo 0
    r = __shfl(r, 0);
  }*/
  //r = curand_uniform(&states[id]);
  r = rands[blockIdx.x];
  //Codemo r del hilo 0
  //r = __shfl(r, 0);
  //if (id % 32 == 0) {printf("Bloque : %d  Hilo : %d   R :%f\n",blockIdx.x, threadIdx.x, r);}
  //printf("Bloque : %d  Hilo : %d   R :%f\n",blockIdx.x, threadIdx.x, r);
  //printf("%f\n", r);
  bool brick = r > value;
  //bool prev = __shfl_up(r, 1);
  //if (id<N)
  //  printf("El hilo %d tiene %d y previo %d con r=%f\n",id, actual, prev, r);
  if(threadIdx.x < n){atomicAdd(&auxiliar, (int)brick);}
  //auxiliar = auxiliar + brick;
  //printf("El hilo %d tiene un %d. r=%f -> El elegido es : %d\n",id, brick,r, auxiliar);
  if (threadIdx.x == 0){
    out[blockIdx.x] = auxiliar;
  }
}

// Segundo espín de la ruleta
__global__ void roulette2(float * probs, int * leaders, int * out, float * rands){
  /* N es el array con los datos de entradoa y N
  el tamaño de dicho array */
  __shared__ int auxiliar;
  auxiliar = 0;
  int id = blockIdx.x * N + leaders[blockIdx.x] * 32 + threadIdx.x;
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
  //if (id < n) {in[id] = value;}

  // Conseguimos un valor aleatorio
  float r;
  /*if (id == 0){
    // Si es el hilo cero lo generamos
    r = curand_uniform(&states[id]);
  } else {
    // si no, se lo copiamos al hilo 0
    r = __shfl(r, 0);
  }*/
  //r = curand_uniform(&states[id]);
  r = rands[blockIdx.x];
  //Codemo r del hilo 0
  //r = __shfl(r, 0);
  //if (id % 32 == 0) {printf("Bloque : %d  Hilo : %d   R :%f\n",blockIdx.x, threadIdx.x, r);}
  //printf("%f\n", r);
  bool brick = r > value;
  //bool prev = __shfl_up(r, 1);
  //if (id<N)
  //  printf("El hilo %d tiene %d y previo %d con r=%f\n",id, actual, prev, r);
  atomicAdd(&auxiliar, (int)brick);
  //printf("1\n");
  //auxiliar = auxiliar + brick;
  //printf("El hilo %d tiene un %d. r=%f -> El elegido es : %d\n",id, brick,r, auxiliar);
  if ((id & 31) == 0){
    //printf("%d\n", leaders[blockIdx.x]);
    //printf("En el bloque %d, auxiliar vale %d, el random %f, y el warp : %d\n", blockIdx.x, auxiliar, r, leaders[blockIdx.x] );
    out[blockIdx.x] = auxiliar + leaders[blockIdx.x] * 32;
  }
}

__global__ void update_paths(int * paths, int * pos_act, int * winners, int * k){
  // lanzamos N hilos
  int id  =  blockIdx.x * blockDim.x + threadIdx.x;
  // copiamos pos_act a paths.
  paths[id * N + *k] = pos_act[id];
  // copiamos winners a pos_act
  pos_act[id] = winners[id];
  // sumamos 1 al contador de pasos, k
  //   cuidado con k, que es un puntero. k++ cambiaría el sitio dónde apunta
  if (threadIdx.x == 0) {*k = *k +1;}
}

__global__ void update_cost(int * paths, float * cost, float * a){
  /* Necesitaremos NxN hilos. Sumando N hilos de forma atómica en cada
  posición del array de costes

  La opción para hacer lo más eficiente es hacer reduce...

  Copiamos a memoria compartida, y leugo reducimos ese array


  Para mejorar la eficiencia lo que vamos a hacer es una combinación de ambos.
  Vamos aplicar reductions a nivel de warp, y luego sumar atómicamente cada
  resultago, generando así muchas menos operaciones atómicas, y muchas más
  en paralelo.*/

  int i = paths[blockIdx.x * blockDim.x + threadIdx.x];
  int j = paths[blockIdx.x * blockDim.x + ((threadIdx.x + 1) % N)];
  //printf("Sumamos A[%d, %d]\n", i, j);
  atomicAdd(&cost[blockIdx.x], a[i * N + j]);
}

__global__ void update_cost2(int * paths, float * cost, float * a){
  /* Necesitaremos NxN hilos. Sumando N hilos de forma atómica en cada
  posición del array de costes

  La opción para hacer lo más eficiente es hacer reduce...

  Copiamos a memoria compartida, y leugo reducimos ese array


  Para mejorar la eficiencia lo que vamos a hacer es una combinación de ambos.
  Vamos aplicar reductions a nivel de warp, y luego sumar atómicamente cada
  resultago, generando así muchas menos operaciones atómicas, y muchas más
  en paralelo.

  Esta versión es 2x más rápida con pocos datos, y 5x cuando nos acercamos a
  1024*/

  float myCost;
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int i = paths[blockIdx.x * blockDim.x + threadIdx.x];
  int j = paths[blockIdx.x * blockDim.x + ((threadIdx.x + 1) % N)];
  //printf("Sumamos A[%d, %d]\n", i, j);

  myCost = a[i * N + j];
  __syncthreads();

  int offset;
  // Realizamos la suma para todos los valores
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    myCost += __shfl_down(myCost, offset);
  }
  // Salvamos únicamente los múltiplos de 32
  if ((id & 31) == 0){
    // Calculamos el hilo dentro del bloque / 32 para ver qué hilo es dentro del warp
    // Lo guardamos en las n-primeras posiciones del array modulado en trozos de 32
    //if (blockIdx.x == 0){printf("= %f\n", value );}
    atomicAdd(&cost[blockIdx.x], myCost);
  }
}



//// Para bsucar el mínimo

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


// Calcula el íncice del elemento mínimo en un array, y lo guarda en iMin
__global__  void arrayReduction(float * array){
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int thisThreadId = id;
  int value = array[id];
  int gap, id2;
  float value2;
  for (gap = WARP_SIZE/2; gap > 0; gap >>= 1){
    id2 = __shfl_down(id, gap);
    value2 = __shfl_down(value, gap);
    //if (threadIdx.x == 0) printf("Gap : %d, value : %d, value2 %d, id : %d, id2 : %d\n", gap, value, value2, id2);
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

__global__ void printiMin(){
  printf("%d\n", iMin);
}


//// Para las feromonas

// Evaporamos las feromonas
__global__ void evaporate(float * ferom){
  // Lanzamos N bloques con N hilos
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("%d\n", id);
  ferom[id] = (1 - RHO) * ferom[id];
}

// Aumentamos la cantidad de feromona del mejor camino
__global__ void deposit_pherom(float * ferom, int * paths, float * cost){
  // lanzamos N hilos
  float coste = cost[iMin];
  float incremento = C / coste;
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  // camino desde :
  int i = paths[iMin * N + id];
  // hasta :
  int j = paths[iMin * N + (id + 1)];
  // Como la matriz es simétrica, sumamos en ambos sentidos
  ferom[i * N + j] += incremento;
  ferom[j * N + i] += incremento;
}

// -- PASOS ALGORITMO

void setStuff(float * Cost, int * P, int * K, int * pos_actual){
  fill<<<1,N>>>(Cost, 0);
  ifill<<<N,N>>>(P, -1);
  init_K<<<1,1>>>(K);
  init_pos_actual<<<1,N>>>(pos_actual);
}

void moveAnts(float * A, float * F, int * P, int * pos_actual, float * Probs, float * Steve, int * wleaders, int * winners, float * rands, int * K){
  for (int i = 0; i < N; i++){
    //----- Mover hormigas
    set_probs<<<N, N>>>(A, F, P, pos_actual, Probs);
    //fcudaPrint(Probs, N*N, N);

    // Rellenamos Steve con las probabilidades acumuladas de los warps
    sum_up<<<N,N>>>(Probs, Steve);
    //fcudaPrint(Steve, N*WARP_SIZE, WARP_SIZE);

    // Llenamos el array de los líderes
    roulette1<<<N,32>>>(Steve, N_WARPS, wleaders, rands);
    //printf("\nLeaders :\n");icudaPrint(wleaders, N, N);
    //icudaPrint(wleaders, N, N)

    roulette2<<<N,32>>>(Probs, wleaders, winners, rands);
    //printf("\nWinners :\n");icudaPrint(winners, N, N);

    update_paths<<<1,N>>>(P, pos_actual, winners, K);
    //printf("\nPos actual, después de actualizar : \n");
    //icudaPrint(pos_actual, N, N);
    //printf("\nY estos son los caminos : \n" );
    //icudaPrint(P, N*N, N);
    //icudaPrint(K,1,1);
  }
}

void findBestAnt(int * P, float * Cost, float * A){
  update_cost2<<<N,N>>>(P, Cost, A);
  //fcudaPrint(Cost, N, N);
  arrayReduction<<<1,N>>>(Cost);
}

void updatePheromoneLevels(float * F, int * P, float * Cost){
  evaporate<<<N,N>>>(F);
  //printf("\n");
  //fcudaPrint(F, N*N, N);
  deposit_pherom<<<1,N>>>(F,P,Cost);
}

__global__ void printSolution(int * paths, float * costs){
  int i;
  printf("\n[");
  for (i = 0; i < N; i++){
    printf(" %d,", paths[iMin * N + i]);
  }
  printf("]\n");
  printf("With a cost : %f\n", costs[iMin]);
}

// Puts the solution on the host
void sol_to_host(float * costs, float * solutions, int n_exp){
  int i;
  cudaMemcpy(&i, &iMin, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&solutions[n_exp], &costs[i], sizeof(float), cudaMemcpyDeviceToHost);
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

// Min from arr
float f_mmin(float * arr, int n){
  float mmin = arr[0];
  int i;
  for (i = 1; i < n; i ++){
    if (arr[i] < mmin){
      mmin = arr[i];
    }
  }
  return mmin;
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


// Std deviation from the n first elements of arr
float fdev(float * arr, float mean, int n){
  float ssum = 0;
  int i;
  for (i = 0; i < n; i++){
    ssum += pow(arr[i] - mean, 2);
  }
  return sqrt(1/(float)(n - 1) * ssum);
}

// The sum of the elements in a float array of lenght n
float fssum(float * arr, int n){
  float ssum = 0;
  int i;
  for (i = 0; i < n; i++){
    ssum += arr[i];
  }
  return ssum;
}

int main(void){

  // Declarations
  float Dist[N*N];
  float * A;

  // Loading from file...
  FILE * my_file;
  my_file = fopen(MAP,"r");

  int i;
  for (i = 0; i < N*N; i++){
    fscanf(my_file, "%f", &Dist[i]);
  }

  fclose(my_file);
  //...

  // Copying to GPU
  cudaMalloc((void **) &A, N * N * sizeof(float));
  cudaMemcpy(A, Dist, N * N * sizeof(float), cudaMemcpyHostToDevice);

  //fcudaPrint(A, N*N, N);


  // Generamos el contador de iteraciones
  int * K;
  cudaMalloc((void **) &K, sizeof(int));
  init_K<<<1,1>>>(K);

  // Generamos los estados
  curandState_t * states;
  cudaMalloc((void**) &states, N * sizeof(curandState_t));
  init_states<<<N,1>>>(time(0), states);

  // Generamos los números aleatorios
  float * rands;
  cudaMalloc((void **) &rands, N * sizeof(float));
  init_rands<<<1,N>>>(rands, states);
  //fcudaPrint(rands, N, N);

  // Generamos F - Feromonas
  float * F;
  cudaMalloc((void **) &F, N * N * sizeof(float));
  fill<<<N,N>>>(F, 1.0);
  //fcudaPrint(F, N * N, N);

  // Generamos P - Paths
  int * P;
  cudaMalloc((void **) &P, N * N * sizeof(int));
  ifill<<<N,N>>>(P, -1);
  //icudaPrint(P, N * N, N );

  // Generamos pos_actual
  int * pos_actual;
  cudaMalloc((void **) &pos_actual, N * sizeof(int));
  init_pos_actual<<<1,N>>>(pos_actual);
  //icudaPrint(pos_actual, N, N);

  // Generamos el array de las probs
  float * Probs;
  cudaMalloc((void **) &Probs, N * N * sizeof(float));
  fill<<<N, N>>>(Probs, 0);

  // Generamos el array de los costes
  float * Cost;
  cudaMalloc((void **) &Cost, N * sizeof(float));
  fill<<<1,N>>>(Cost, 0);

  // Creamos steve
  float * Steve;
  cudaMalloc((void **) &Steve, N * 32 * sizeof(float));
  fill<<<N,32>>>(Steve, 0); // Esto no hace falta, pero lo ponemos para depurar mejor
  //fcudaPrint(Steve, N*32, 32);

  // Creamos el array de warps lídiferentes
  int * wleaders;
  cudaMalloc((void **) &wleaders, N * sizeof(int));
  ifill<<<N,32>>>(wleaders, -1); // Tampoco hace falta, pero...

  // Creamos el array de ciudades ganadoras
  int * winners;
  cudaMalloc((void **) &winners, N * sizeof(int));
  ifill<<<N,32>>>(winners, 1); // Este tampoco hace falta


  // An array in order to save all solutions and timings
  float solutions[N_EXPERIMENTS];
  float times[N_EXPERIMENTS];

  clock_t start, end;
  cudaDeviceSynchronize();  //<< último añadido!!
  int experiment;
  for (experiment = 0; experiment < N_EXPERIMENTS; experiment++){

    start = clock();
    int iters;
    for (iters = 0; iters < N_ITERS; iters++){
      setStuff(Cost, P, K, pos_actual);
      moveAnts(A, F, P, pos_actual, Probs, Steve, wleaders, winners, rands, K);
      findBestAnt(P, Cost, A);
      updatePheromoneLevels(F,P,Cost);
    }
    end = clock();
    sol_to_host(Cost, solutions, experiment);
    times[experiment] = ((float) (end - start)) / CLOCKS_PER_SEC;

  }


  // Summing up
  float min_cost = f_mmin(solutions, N_EXPERIMENTS);
  float mean_cost = fmean(solutions, N_EXPERIMENTS);
  float dev_cost = fdev(solutions, mean_cost, N_EXPERIMENTS);
  float total_time = fssum(times, N_EXPERIMENTS);

  printf("\n--- ACO : Test info ---\n");
  printf("[-] Cities               : %d\n", N);
  printf("[-] Num. max iters       : %d\n", N_ITERS);
  printf("[-] Num. experiments     : %d\n", N_EXPERIMENTS);
  printf("[-] Required time        : %f\n", total_time);
  printf("[-] Required time mean   : %f\n", total_time/N_EXPERIMENTS);


  printf("\n--- Results ---\n");
  printf("[-] Best solution : %f\n", min_cost);
  printf("[-] Solution info :\n");
  printf("[-]    Mean : %f\n", mean_cost);
  printf("[-]    Dev  : %f\n", dev_cost);

  cudaDeviceSynchronize();
  return 0;
}
