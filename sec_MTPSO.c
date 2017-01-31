
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define W 2
#define C1 2
#define C2 2
#define Vs 0.1


#define N_PARTS 128
#define N_COL 3
#define N_ITERS 200
#define N_EXPERIMENTS 10

//#define N 16
//#define GRAPH "graphs/3Graph_16_0.txt"
#define N 32
#define GRAPH "graphs/3Graph_32_0.txt"
//#define N 80
//#define GRAPH "graphs/3Graph_40_1.txt"
//#define N 128
//#define GRAPH "graphs/3Graph_128_0.txt"


// Prints an array of floats
void fMatrixPrint(float * array, int elements, int n_jump){
  int i;
  for (i = 0; i < elements; i++){
      if ((i % n_jump) == 0 ) {printf("\n");}
      printf("%f ", array[i]);
  }
}

// Prints an array of floats
void iMatrixPrint(int * array, int elements, int n_jump){
  int i;
  for (i = 0; i < elements; i++){
      if ((i % n_jump) == 0 ) {printf("\n");}
      printf("%d ", array[i]);
  }
}

// Returns a random float between 0 and 1;
float rand01(){
  float r = rand();
  return (float) r/ (float) RAND_MAX;
}

// Returns a random float between a low and up;
float bounded_rand(float low, float up){
  float r = rand01();
  return (up-low)*r + low;
}

// Returns a random integer between 0 and Matrix
int irand(int max){
  return rand() % max;
}

// Sums all the elements in an integer array
int isum(int * arr, int n){
  int sum = 0;
  int i;
  for (i = 0; i < n; i++){
    sum += arr[i];
  }
  return sum;
}

// Copies the all elements between a position and b position from arr1 to arr2
void iCopyArray(int * arr1, int * arr2, int a, int b){
  int i;
  for (i = a; i < b; i++){
    arr2[i] = arr1[i];
  }
}

// Copies N elements from arr1 to arr2, starting at a2 position
void iCopyOtherArray(int * arr1, int * arr2, int a1, int a2, int n){
  /* Arrays may not have the same number of elements */
  int i;
  for (i = 0; i < N; i++){
    arr2[a2 + i] = arr1[a1 + i];
  }
}

// Fills an array with n zeros
void zeros(int * arr, int n){
  int i;
  for (i=0 ; i <n; i++){
    arr[i] = 0;
  }
}

// Initializes A if you want to make some test using C_n graphs
void init_A(int * a){
  int i, j;
  for (i = 0; i < N; i++){
    for ( j = 0; j < N; j ++){
      if ((i != j) && ((abs(i - j)==1))){
        a[i * N + j] = 1;
      } else {
        a[i * N + j] = 0;
      }
    }
  }
}

// Inits positions randomly
void init_Pos(int * pos){
  int i, j;
  for (i = 0; i < N_PARTS; i++){
    for ( j = 0; j < N; j ++){
      pos[i * N + j] = (int) rand() % N_COL;
    }
  }
}

// Inits velocities randomly
void init_Vel(float * vel){
  int i, j;
  for (i = 0; i < N_PARTS; i++){
    for (j = 0; j < N; j++){
      vel[i * N + j] = bounded_rand(-N_COL, N_COL);
    }
  }
}

// Given a coloring an a graph matrix, builds the conflict matrix
void conflict_matrix(int * coloring, int * a, int * co){
  /* A coloring is a secuence of colors of length N */
  int i, j, color;
  for (i = 0; i < N; i++){
    color = coloring[i];
    for (j = 0; j < N; j++){
      if ((i != j) && (coloring[j] == color)){
        co[i * N + j] = a[i * N + j];
      } else {
        co[i * N + j] = 0;
      }
    }
  }
}

// Given a coloring, computes its fitness
int fitness(int * coloring, int * a){
  int * co, output;
  co = (int *) malloc(N*N*sizeof(int));
  conflict_matrix(coloring, a, co);
  output = isum(co, N*N);
  free(co);
  return output;
}

// -------------- Walking one Strategy ------------
// Computes the number of conflicts of a node j
int Cr(int j, int * co){
  // j : node index
  //co : conflic matrix
  return isum(&co[j*N], N);
}

// Computes the collision factor from Cr
float Cf(int cr){
  // cr : number of conflicts of a node in a concrete coloring
  return 1/(1 + exp((float)(-cr + 2)));
}

// Sigmoid function.
float S(float x){
  // x : Real number
  return (1/(1 + exp(-x)));
}

// Turns velocity into discrete value :  |R --> Z2
int M(float Vj, float Cfj){
  // Vj  :  j-component of particle's velocity
  // Cfj :  collision factor from j-node
  float output;
  output = 0;
  float r = rand01();
  if ((Cfj > r) && (S(Vj) > r)){
    output = 1;
  }
  return output;
}

// Applies walking one strategie to j-node of a particle
void move_node(int * pos, int j, float * v, int * co){
  // Pos : particle position. Usually &Pos[j]
  // j   : node to move
  // v   : particle vel. Usually &Vel[j]
  // co  : conflict matrix
    float Cfj, Vj;
    int MVj;
    Cfj = Cf(Cr(j, co));
    Vj = v[j];
    MVj = M(Vj, Cfj);
    pos[j] = (pos[j] + MVj) % N_COL;
}

// Apllies walking one strategy to a particle
void move_particle(int * pos, float * vel, int * a){
  // pos : Particle position, &Pos[j]
  // vel : particle vel. Usually &Vel[j]
  // a   : graph matrix
  int * co;
  co = malloc(N * N * sizeof(int));
  int i;
  for (i = 0; i < N; i++){
    conflict_matrix(pos, a, co);
    move_node(pos, i, vel, co);
  }
  free(co);
  co = NULL;
}

// Updates al the local mins
void update_min(int * mmin, int * pos, int * a){
  // mmin   : local min's array
  // pos    : position array
  // a      : graph matrix
  int i, f_n, f_o;
  for (i = 0; i < N_PARTS; i ++){
    f_n = fitness(&pos[i*N], a);
    f_o = fitness(&mmin[i*N], a);
    if (f_n < f_o){
      // Cuidado, que iCopy copia un fragmento del array, así que hay que pasárselo entero.
      iCopyArray(pos, mmin, i*N, (i + 1) * N);
    }
  }
}

// Updates the global min
void update_gMin(int * gMin, int * mmin, int * a){
  // gMin  : global min array
  // mmin  : local min's array
  // a     : graph matrix
  int i, f_n, f_o;
  for (i = 0; i < N_PARTS; i++){
    f_n = fitness(&mmin[i*N], a);
    f_o = fitness(gMin, a);
    if (f_n < f_o){
      iCopyOtherArray(mmin, gMin, i*N, 0, N);
    }
  }
}

// ------------------ Turbulent strategy ------------

// Updates the vel of all particles
void update_vel(float * vel, int * pos, int * mmin, int * gmin){
  // vel   : Velocity array
  // pos   : position array
  // mmin  : local min array
  // gmin  : global min array
  int i;
  for (i = 0; i < N*N_PARTS; i++){
    vel[i] = vel[i] * W + C1 * rand01() * (mmin[i] - pos[i]) + C2 * rand01() * (*gmin - pos[i]);
  }
}

// Applies turbulent strategy
void turbulence(float * vel){
  // vel : Velocity array
  int i;
  for (i = 0; i < N * N_PARTS; i ++){
    if (abs(vel[i]) < Vs){
      vel[i] = - rand01();
    }
  }
}

// --------------- Assessment strategy ---------------

// Computes the number of conflicts of each node in a coloration
void conflict_nodes(int * coloring, int * a, int * conflict_list){
  // coloring :  coloring array
  // a        :  graph matrix
  // conflict_list : list integer where we'll put the result (n-sized)
  int * co;
  co = malloc(N * N * sizeof(int));
  conflict_matrix(coloring, a, co);
  int i, j;
  for (i = 0 ; i < N; i++){
    conflict_list[i] = 0;
    for (j = 0; j < N; j++){
      conflict_list[i] += co[i*N + j];
    }
  }
}

// Returns the maximum conflicting node in a coloration
int max_conflict_node(int * coloring, int * a){
  // coloring  : coloring array
  // a         : graph matrix
  int i_max = 0;
  int * conf_nodes;
  conf_nodes = malloc(N * sizeof(int));
  conflict_nodes(coloring, a, conf_nodes);
  int i;
  for (i = 0; i < N; i++){
    if (conf_nodes[i] > conf_nodes[i_max]){
      i_max = i;
    }
  }
  free(conf_nodes);
  conf_nodes = NULL;
  return i_max;
}


int assessment(int max_confl_node, int i_part, int * pos, int * a){
  /*
  max_confl_node es el nodo con más conflictos
  i_part es el índice de la partícula
  pos es el array de las posiciones,
  a es la matriz de adyacencia

  devuelve un bool diciendo si se ha cambiado el nodo con más
  conflictos*/
  int changed = 0;
  int color = 0;
  int * coloring, f_n, f_o;
  coloring = malloc(N * sizeof(int));
  while ((color < N_COL) && (!changed)){
    iCopyOtherArray(pos, coloring, i_part * N, 0, N);
    coloring[max_confl_node] = color;
    f_n = fitness(coloring, a);
    f_o = fitness(&pos[i_part * N], a);
    if (f_n < f_o){
      changed = 1;
      iCopyOtherArray(coloring, pos, 0, i_part * N, N);
    }
    color += 1;
  }
  return changed;
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





int main(void){
  srand (time(NULL));

  // Declarations
  int A[N*N];

  // Loading from file...
  FILE * my_file;
  my_file = fopen(GRAPH,"r");
  int k;
  for (k = 0; k < N*N; k++){
    //printf("%d\n", k);
    fscanf(my_file, "%d", &A[k]);

  }
  fclose(my_file);

  int Pos[N*N_PARTS];
  int MMin[N*N_PARTS];
  int gMin[N];
  float Vel[N*N_PARTS];
  int co[N*N];

  // Some thins only to save results
  int solutions[N_EXPERIMENTS];
  int stops[N_EXPERIMENTS];

  clock_t start, end;
  double cpu_time_used;

  start = clock();

  int experiment;
  for (experiment = 0; experiment < N_EXPERIMENTS; experiment ++){

    // Initialising things
    init_Pos(Pos);
    init_Vel(Vel);
    zeros(MMin, N*N_PARTS);
    zeros(gMin, N);


    int iter, i, j;
    while (iter < N_ITERS && fitness(gMin, A) > 0) {
      update_min(MMin, Pos, A);
      update_gMin(gMin, MMin, A);
      update_vel(Vel, Pos, MMin, gMin);
      for (j = 0; j < N_PARTS; j++){
        int cambiado = assessment(max_conflict_node(&Pos[j*N], A), j, Pos, A);
        if (!cambiado){
          move_particle(Pos, Vel, A);
        }
      }
      iter ++;
    }
    solutions[experiment] = fitness(gMin, A);
    stops[experiment] = iter;
  }


  printf("\n--- MTPSO : Test info ---\n");
  printf("[-] Population       : %d\n", N_PARTS);
  printf("[-] Variables        : %d\n", N);
  printf("[-] Num. max iters   : %d\n", N_ITERS);
  printf("[-] Num. experiments : %d\n", N_EXPERIMENTS);

  int sol_mean = imean(solutions, N_EXPERIMENTS);
  int iter_mean = imean(stops, N_EXPERIMENTS);

  printf("\n--- Results ---\n");
  printf("[-] Best solution : %d\n", immin(solutions, N_EXPERIMENTS));
  printf("[-]  Solution info :\n");
  printf("      Mean : %d\n", sol_mean);
  printf("      Dev  : %f\n", idev(solutions, imean(solutions, N_EXPERIMENTS), N_EXPERIMENTS));
  printf("[-]  Max iter info :\n");
  printf("      Mean : %d \n", iter_mean);
  printf("      Dev  : %f\n", idev(stops, imean(stops, N_EXPERIMENTS), N_EXPERIMENTS));


  int colorable = 0;
  int i;
  for(i = 0; i < N_EXPERIMENTS; i++){
    if (solutions[i]==0){
      colorable = 1;
    }
  }
  if(colorable){
    printf("\n[-]  Graph is 3-colorable!\n\n");
  }

  return 0;
}
