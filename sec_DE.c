/*

FUNCTION
---------------------
    SPH . Sphere.               Min : x_i = 1                     Min Value : 0                    Bounds : [-10, 10]
    SKT . Styblinski-Tang       Min : x_i = -2.903534             Min Value : -39.16599 * N        Bounds : [-5 , 5 ]
    DXP . Dixon-Price           Min : x_i = 2^(-(2^i - 2)/(2^2))  Min Value : 0                    Bounds : [-10, 10]
    RSB . Rosenbrock            Min : x_i = 1                     Min Value : 0                    Bounds : [-2.048, 2.048]
    ZKV . Zakharov              Min : x_i = 0                     Min Value : 0                    Bounds : [-5, 10]
    ACK . Ackley                Min : x_i = 0                     Min Value : 0                    Bounds : [-32.768, 32.768]
    GWK . Griewank              Min : x_i = 0                     Min Value : 0                    Bounds : [-600, 600]
    RTG . Rastrigin             Min : x_i = 0                     Min Value : 0                    Bounds : [-5.12, 5.12]
    LEV . Levy                  Min : x_i = 1                     Min Value : 0                    Bounds : [-10, 10]
*/


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>


// Problem settings
#define F_KEY SPH // Benchmark function from above
#define N 30   // Dimension
#define POPULATION 128 // Population
#define UB 10 // Upper bound
#define LB -10 // Lower bound

// Control parameters
#define CR 0.8
#define F 0.5

// Experiment settings
#define N_ITERS 1000 // Max iters
#define N_EXPERIMENTS 2
#define TOLERANCE 1E-6


// Prints an array of floats
void fMatrixPrint(float * array, int elements, int n_jump){
  int i;
  for (i = 0; i < elements; i++){
      if ((i % n_jump) == 0 ) {printf("\n");}
      printf("%f ", array[i]);
  }
}

// Prints an array of integers
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

int ibounded_rand(float low, float up){
  return (int) bounded_rand(low, up);
}


// Función a optimizar : Sphere
float SPH(float * x){
  float output, sum;
  int i, j;

  sum = 0;
  for (i = 0; i < N; i ++){sum += (x[i]-1) * (x[i]-1);}
  output = sum;

  return output;
}

// Función a optimizar : Styblinski-Tang
float SKT(float * x){
  float output, sum;
  int i, j;

  sum = 39.16599 * N;
  for (i = 0; i < N; i ++){sum += 0.5*pow(x[i],4) - 8*pow(x[i],2) + 2.5*x[i];}
  output = sum;

  return output;
}

// Función a optimizar : Dixon-Price
float DXP(float * x){
  float output, sum;
  int i, j;

  sum = pow((x[0] - 1),2);
  for (i = 1; i < N; i ++){sum += i * pow(2*pow(x[i],2) - x[i-1], 2);}
  output = sum;

  return output;
}

// Función a optimizar : Rosenbrock
float RSB(float * x){
  float output, sum;
  int i, j;

  sum = 0;
  for (i = 0; i < N-1; i ++){sum += 100 * pow(x[i+1] - pow(x[i],2),2) + pow(x[i] - 1, 2);}
  output = sum;

  return output;
}

// Función a optimizar : Zakharov
float ZKV(float * x){
  float output, sum;
  int i, j;

  float sum0 = 0;
  float sum1 = 0;
  for (i = 0; i < N; i ++){
    sum0 += pow(x[i],2);
    sum1 += 0.5 * i * x[i];
  }
  output = sum0 + pow(sum1,2) + pow(sum1,4);


  return output;
}

// Función a optimizar : Ackley
float ACK(float * x){
  float output, sum;
  int i, j;


  float sum0 = 0;
  float sum1 = 0;
  float a = 20;
  float b = 0.2;
  float c = 2 * M_PI;
  for (i = 0; i < N; i ++){
    sum0 += 1/((float) N) * pow(x[i], 2);
    sum1 += 1/((float) N) * cos(c * x[i]);
  }
  output = -a * exp(-b * sqrt(sum0)) - exp(sum1) + a + exp(1);

  return output;
}

// Función a optimizar : Griewank
float GWK(float * x){
  float output, sum;
  int i, j;

  sum = 0;
  float prod = 1;
  for (i = 0; i < N; i ++){
    sum += pow(x[i],2)/4000.0;
    prod = prod * cos(x[i]/sqrt(i+1));
  }
  output = sum - prod + 1;

  return output;
}

// Función a optimizar : Rastrigin
float RTG(float * x){
  float output, sum;
  int i, j;

  sum = 10 * N;
  for (i = 0; i < N; i ++){sum += pow(x[i],2) - 10*cos(2 * M_PI * x[i]);}
  output = sum;

  return output;
}

// Función a optimizar : Levy
float LEV(float * x){
  float output, sum, wi;
  int i, j;

  sum = 0;

  for (i = 0; i < N; i++){
    if (i == 0){
      sum += pow(sin(M_PI * (1 + (x[i] - 1)/4)),2) + pow((1 + (x[i] - 1)/4) - 1, 2) * (1 + 10 * pow(sin(M_PI * (1 + (x[i] - 1)/4) + 1),2));
    } else if (i == N - 1) {
      sum += pow((1 + (x[i] - 1)/4) - 1, 2) * (1 + pow(sin(2 * M_PI * (1 + (x[i] - 1)/4)),2));
    } else {
      sum += pow((1 + (x[i] - 1)/4) - 1, 2) * (1 + 10 * pow(sin(M_PI * (1 + (x[i] - 1)/4) + 1),2));
    }
  }
  output = sum;

  return output;
}

// Init population
void init_pos(float * pos){
  int i, j;
  for (i = 0; i < POPULATION; i ++){
    for (j = 0; j < N; j++){
      pos[i * N + j] = bounded_rand(LB, UB);
    }
  }
}

void init_gMin(float * gmin){
  int i;
  for (i = 0; i < N; i++){
    gmin[i] = bounded_rand(LB, UB);
  }
}

// Mute one single member of the population
void mutation(float * pos, float * new_pos, int x, int a, int b, int c){
  int R = ibounded_rand(0,N);
  float r;
  int i;
  for (i = 0; i < N; i ++){
    r = rand01();
    if (r < CR || i == R){
      new_pos[x * N + i] = pos[a * N + i] + F * (pos[b * N + i] - pos[c * N + i]);
    } else {
      new_pos[x * N + i] = pos[x * N + i];
    }
  }
}

// Mute all the population
void mute_population(float * pos, float * new_pos){
  int x, a, b, c;
  for (x = 0; x < POPULATION; x++){
    a = ibounded_rand(0, POPULATION);
    b = ibounded_rand(0, POPULATION);
    c = ibounded_rand(0, POPULATION);
    mutation(pos, new_pos, x, a, b, c);
  }
}

// Keeps the best members of the population in memory
void choose_best_members(float * pos, float * new_pos, float * gmin){
  int i,j;
  float old, new;
  for (i = 0; i < POPULATION; i++){
    old = F_KEY(&pos[i*N]);
    new = F_KEY(&new_pos[i*N]);
    if (new < old){
      // Keep the new position
      for (j = 0; j < N; j++) {
        pos[i * N + j] = new_pos[i * N + j];
      }
      // Check if is a glomal min.
      if (new < F_KEY(gmin)){
        // If it's a global min, then keep it
        for (j = 0; j < N; j++) {
          gmin[j] = new_pos[i * N + j];
        }
      }
    }
  }
}

void combine(float * pos, float * new_pos, float * gmin){
  mute_population(pos, new_pos);
  choose_best_members(pos, new_pos, gmin);
}

void printSolution(float * sol){
  int i;
  for ( i = 0; i < N; i++){
    printf("%f ", sol[i]);
  }
  printf("\n");
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

// Return the integer mean of the n first elements from arr
int imean(int * arr, int n){
  int ssum = 0;
  int i;
  for (i = 0; i < n; i++){
    ssum += arr[i];
  }
  return ssum/n;
}

// Min from arr
float fmmin(float * arr, int n){
  float mmin = arr[0];
  int i;
  for (i = 1; i < n; i ++){
    if (arr[i] < mmin){
      mmin = arr[i];
    }
  }
  return mmin;
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

  // Init the stuff
  float * pos;
  float * new_pos;
  float * gMin;

  pos = (float *) malloc(N * POPULATION * sizeof(float));
  new_pos = (float *) malloc(N * POPULATION * sizeof(float));
  gMin = (float *) malloc(N * sizeof(float));

  // Some thins only to save results
  float solutions[N_EXPERIMENTS];
  int stops[N_EXPERIMENTS];

  clock_t start, end;
  double cpu_time_used;

  start = clock();


  int experiment;
  for (experiment = 0; experiment < N_EXPERIMENTS; experiment ++){


    //------------------------- DE Algorithm -----------------------------------

    // Init things
    init_pos(pos);
    init_gMin(gMin);

    int iter = 0;
    while (iter < N_ITERS && F_KEY(gMin) > TOLERANCE) {
      combine(pos, new_pos, gMin);
      iter ++;
    }
    //--------------------------------------------------------------------------

    solutions[experiment] = F_KEY(gMin);
    stops[experiment] = iter;

  }

  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;


  printf("\n--- DE : Test info ---\n");
  printf("[-] Population       : %d\n", POPULATION);
  printf("[-] Variables        : %d\n", N);
  printf("[-] Num. max iters   : %d\n", N_ITERS);
  printf("[-] Num. experiments : %d\n", N_EXPERIMENTS);
  printf("[-] Required time        : %f\n", cpu_time_used);
  printf("[-] Required time mean   : %f\n", cpu_time_used/N_EXPERIMENTS);

  float sol_mean = fmean(solutions, N_EXPERIMENTS);
  int iter_mean = imean(stops, N_EXPERIMENTS);

  printf("\n--- Results ---\n");
  printf("[-] Best solution : %f\n", fmmin(solutions, N_EXPERIMENTS));
  printf("[-] Solution info :\n");
  printf("[-]    Mean : %f\n", sol_mean);
  printf("[-]    Dev  : %f\n", fdev(solutions, fmean(solutions, N_EXPERIMENTS), N_EXPERIMENTS));
  printf("[-] Max iter info :\n");
  printf("[-]    Mean : %d \n", iter_mean);
  printf("[-]    Dev  : %f \n\n", idev(stops, imean(stops, N_EXPERIMENTS), N_EXPERIMENTS));
  return 0;
}
