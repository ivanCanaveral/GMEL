#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define N 14
#define MAP "maps/14Burma.txt"
//#define N 16
//#define MAP "maps/16Ulysses.txt"
//#define N 24
//#define MAP "maps/24Liss.txt"
//#define N 48
//#define MAP "maps/48att.txt"
//#define N 52
//#define MAP "maps/52Berlin.txt"
//#define N 76
//#define MAP "maps/76Pr.txt"
//#define N 96
//#define MAP "maps/96gr.txt"
//#define N 130
//#define MAP "maps/130Ch.txt"
//#define N 137
//#define MAP "maps/137gr.txt"
//#define N 264
//#define MAP "maps/264pr.txt"


#define N_EXPERIMENTS 10
#define N_ITERS 300

#define ALPHA 1
#define BETA 2
#define RHO 0.5
#define C N/8

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

// Genera un valor aleatorio entre 0 y 1;
float rand01(){
  float r = rand();
  return (float) r/ (float) RAND_MAX;
}

// Genera un valor aleatorio entre dos dados;
float bounded_rand(float low, float up){
  float r = rand01();
  return (up-low)*r + low;
}

// Returns a random integer between 0 and Matrix
int irand(int max){
  return rand() % max;
}


void init_A(float * a){
  int i, j;
  int id;
  for (i = 0; i < N; i ++){
    for (j = 0; j < N; j ++){
      id = i * N + j;
      if (i==j){
        a[id] = 999999;
      } else if ((abs(i-j)==1) || (i == 0 && j == N-1) || (j == 0 && i == N-1)) {
        a[id] = 1;
      } else {
        a[id] = 100;
      }
      //a[id] = 3;
    }
  }
}

// Llena un array de floats de tamaño n con un número x
void ffill(float * arr, int n, float x){
  int i;
  for (i = 0; i < n; i++){
    arr[i] = x;
  }
}

// Llena un array de floats de tamaño n con un número x
void ifill(int * arr, int n, int x){
  int i;
  for (i = 0; i < n; i++){
    arr[i] = x;
    //printf("%d\n",i);
  }
}

// Initialises path's Matrix
void init_paths(int * paths){
  // Here is where we choose fill all cities, or make random elections for initial cities
  ifill(paths, N * N, -1);
  int ant;
  for (ant = 0; ant < N; ant++){
    paths[ant * N] = ant; // Each ant starts its own citie
    paths[ant * N] = irand(N); // Each ant starts from a random city
  }
}

// Returns 1 if a city has been visited yet
int is_in_path(int ciudad, int * camino){
  int i;
  int encontrado = 0;
  for (i = 0; i < N; i ++){
    if (camino[i] == ciudad){
      encontrado = 1;
    }
  }
  return encontrado;
}

// Computes the probabilities of the next move for a given ants
void P(float * probs, int * paths, int ant, int step, float * dist, float * pherom){
  int origin = paths[ant * N + step];

  int destination;
  for (destination = 0; destination < N; destination ++){
    if (is_in_path(destination, &paths[ant * N])) {
      probs[destination] = 0;
    } else {
      probs[destination] = pow(1/dist[origin * N + destination], BETA) * pow(pherom[origin * N + destination], ALPHA);
    }
  }
}

// Normalizes an array of floats
void normalize(float * arr, int size){
  float ssum = 0;
  int i;
  for (i = 0; i < N; i ++){
    ssum += arr[i];
    arr[i] = ssum;
  }
  float mmax = arr[N-1];
  for (i = 0; i < N; i ++){
    arr[i] = arr[i] / mmax;
  }
}

// Returns an idex with probability from probs
int roulette(float * probs){
  int i = 0;
  normalize(probs, N);
  float r = rand01();
  //printf("%f\n", r);
  while (probs[i] < r){
    i += 1;
  }
  return i;
}

// Builds a complete tour for a single ant
void build_path(float * dist, float * pherom, int * paths, float * probs, int ant){
  int step;
  for (step = 0; step < N-1; step++){
    P(probs, paths, ant, step, dist, pherom);
    paths[ant * N + step + 1] = roulette(probs);
  }
}

// Builds a tour for all ants
void build_paths(float * dist, float * pherom, int * paths, float * probs){
  int ant;
  for (ant = 0; ant < N; ant++){
    build_path(dist, pherom, paths, probs, ant);
  }
}

// Computes the cost of each path
void compute_cost(int * paths, float * dist, float * costs){
  int ant, step, origin, destination;
  float cost;
  for (ant = 0; ant < N; ant++){
    cost = 0;
    for (step = 0; step < N; step++){
      origin = paths[ant * N + step];
      destination = paths[ant * N + (step + 1) % N];
      cost += dist[origin * N + destination];
    }
    costs[ant] = cost;
  }
}

// Computes the cost of one path
float compute_path_cost(int * path, float * dist){
  int step, origin, destination;
  float cost;

  cost = 0;
  for (step = 0; step < N; step++){
    origin = path[step];
    destination = path[(step + 1) % N];
    cost += dist[origin * N + destination];
  }
  return cost;
}

// Evaporates pheromon levels
void evaporate(float * pherom){
  int i, j;
  for (i = 0; i < N; i++){
    for (j = 0; j < N; j++){
      pherom[i * N + j] = (1 - RHO) * pherom[i * N + j];
    }
  }
}

// Deposits pheromones
void deposit_pherom(float * pherom, int * paths, float * costs){
  int step, ant, origin, destination;
  float amount;
  for (ant = 0; ant < N; ant++){
    amount = C / costs[ant];
    //printf("%f ,  %f,  %f\n", C, amount, costs[ant]);
    for (step = 0; step < N; step++){
      origin = paths[ant * N + step];
      destination = paths[ant * N + (step + 1) % N];
      pherom[origin * N + destination] += amount;
      pherom[destination * N + origin] += amount; //Symmetric problem
    }
  }
}

// Evaporates and deposti pherom
void update_pherom(float * pherom, int * paths, float * costs){
  evaporate(pherom);
  deposit_pherom(pherom, paths, costs);
}


// Looks for the minimun elemnt in an array
int fmmin(float * arr, int size){
  int mmin = 0;
  int i;
  for (i = 1; i < size; i++){
    if (arr[i] < arr[mmin]){
      mmin = i;
    }
  }
  return mmin;
}

// Updates the best_tour values
void update_best_tour(int * best, float * best_cost, float * costs, int  * paths){
  int ant, i;
  for (ant = 0; ant < N; ant++){
    if (costs[ant] < best_cost[0]){
      //printf("asdfadf\n");
      best_cost[0] = costs[ant];
      for (i = 0; i < N; i++){
        best[i] = paths[ant * N + i];
      }
    }
  }
}

// Imprime un array de floats
void printSol(int * best_sol, float * best_cost){
  int i;
  printf("[");
  for (i = 0; i < N-1; i++){
      printf("%d, ", best_sol[i]);
  }
  printf("%d]\n", best_sol[N-1]);
  printf("Coste %f\n", best_cost[0]);
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

int main(void){
  srand (time(NULL));

  // Declarations
  float Dist[N*N];
  float Pherom[N*N]; // Pheromone Matrix
  float Probs[N]; // Probabilities
  float Cost[N]; // Costs of each path
  int Paths[N*N]; // Paths of all ants

  float solutions[N_EXPERIMENTS];

  int best_tour[N];
  float best_tour_cost[2];

  // Loading from file...

  FILE * my_file;
  my_file = fopen(MAP,"r");

  int i;
  for (i = 0; i < N*N; i++){
    fscanf(my_file, "%f", &Dist[i]);
  }

  fclose(my_file);
  //...

  clock_t start, end;
  double cpu_time_used;

  start = clock();

  int experiment;
  for (experiment = 0; experiment < N_EXPERIMENTS; experiment++){
    // Init Pheromone leves
    ffill(Pherom, N*N, 0.5);
    // Init best_tour_cost
    best_tour_cost[0] = 9999999;

    int iter;
    for (iter = 0; iter < N_ITERS; iter++){
      init_paths(Paths);
      build_paths(Dist, Pherom, Paths, Probs);
      compute_cost(Paths, Dist, Cost);
      update_pherom(Pherom, Paths, Cost);
      update_best_tour(best_tour, best_tour_cost, Cost, Paths);
      //if (iter % 100 == 0){
        //printf("____Parcial iteración Nº : %d\n", iter);
        //printSol(best_tour, best_tour_cost);
      //}
    }
    //printf("\n__Definitiva Experimento Nº : %d\n", experiment);
    //printSol(best_tour, best_tour_cost);
    //printf("\n\n\n\n");
    solutions[experiment] = best_tour_cost[0];
  }

  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

  // Summing up
  float min_cost = f_mmin(solutions, N_EXPERIMENTS);
  float mean_cost = fmean(solutions, N_EXPERIMENTS);
  float dev_cost = fdev(solutions, mean_cost, N_EXPERIMENTS);

  printf("\n--- ACO : Test info ---\n");
  printf("[-] Cities               : %d\n", N);
  printf("[-] Num. max iters       : %d\n", N_ITERS);
  printf("[-] Num. experiments     : %d\n", N_EXPERIMENTS);
  printf("[-] Required time        : %f\n", cpu_time_used);
  printf("[-] Required time mean   : %f\n", cpu_time_used/N_EXPERIMENTS);


  printf("\n--- Results ---\n");
  printf("[-] Best solution : %f\n", min_cost);
  printf("[-] Solution info :\n");
  printf("[-]    Mean : %f\n", mean_cost);
  printf("[-]    Dev  : %f\n", dev_cost);

  //fMatrixPrint(solutions, N_EXPERIMENTS, N_EXPERIMENTS);
  return 0;
}
