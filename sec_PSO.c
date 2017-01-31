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
#define W 0.6
#define PHI_P 1.6
#define PHI_G 1.6

// Experiment settings
#define N_ITERS 1000 // Max iters
#define N_EXPERIMENTS 100
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

// Generates a random value between 0 and 1 U(0,1);
float rand01(){
  float r = rand();
  return (float) r/ (float) RAND_MAX;
}

// Returns a random value between two given numbers U(low, up);
float bounded_rand(float low, float up){
  float r = rand01();
  return (up-low)*r + low;
}

// Returns a random integer bewteen two gien numbers
int ibounded_rand(float low, float up){
  return (int) bounded_rand(low, up);
}

// Función a optimizar : Sphere
float SPH(float * x){
  float output, sum;
  int i, j;

  sum = 0;
  for (i = 0; i < N; i ++){
    //printf("pos: %d   val: %f   result: %f \n", i, x[i], (x[i]-1) * (x[i]-1));
    sum += (x[i]-1) * (x[i]-1);
  }
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

// Init vel
void init_vel(float * vel){
  int i, j;
  for (i = 0; i < POPULATION; i ++){
    for (j = 0; j < N; j++){
      vel[i * N + j] = bounded_rand(LB, UB);
    }
  }
}

// Init the min array with zeros
void init_min(float * min){
  int i, j;
  for (i = 0; i < POPULATION; i ++){
    for (j = 0; j < N; j++){
      min[i * N + j] = 0;
    }
  }
}

// Init the min_evals with high values in order to guarantee the change
void init_min_evals(float * min_evals){
  int i, j;
  for (i = 0; i < POPULATION; i ++){
    for (j = 0; j < N; j++){
      min_evals[i * N + j] = 99999;
    }
  }
}

// Init the global min ramdomly
void init_gMin(float * gmin){
  int i;
  for (i = 0; i < N; i++){
    gmin[i] = bounded_rand(LB, UB);
  }
}

// Evaluates the function in each particle
void eval(float * pos, float * evals){
  int i;
  for (i = 0; i < POPULATION; i++){
    //printf("%f\n", F_KEY(&pos[i * N]));
    evals[i] = F_KEY(&pos[i * N]);
  }
}

// Updates the min array
void update_min(float * pos, float * min, float * evals, float * min_evals){
  int i, j;
  for (i = 0; i < POPULATION; i++){
    if (evals[i] < min_evals[i]){
      min_evals[i] = evals[i];
      for (j = 0; j < N; j++){
        min[i * N + j] = pos[i * N + j];
      }
    }
  }
}

// Updates gMin
void update_gMin(float * min, float * min_evals, float * gMin){
  int i,j;
  for (i = 0; i < POPULATION; i++){
    //printf("%f vs %f\n", min_evals[i], F_KEY(gMin));
    if (min_evals[i] < F_KEY(gMin)){
      //printf("Yeah\n");
      for (j = 0; j < N; j++){
        gMin[j] = min[i * N + j];
      }
    }
  }
}

// Compares all the info of the particles in order to get the best solutions
void find_mins(float * pos, float * min, float * evals, float * min_evals, float * gMin){
  // Find out how good the particles are
  eval(pos, evals);
  // Compare new vs old results
  update_min(pos, min, evals, min_evals);
  // Find the best solution so far
  update_gMin(min, min_evals, gMin);
  //fMatrixPrint(gMin, N, N);
}

// Updates the vel of each Particle
void update_vel(float * pos, float * vel, float * min, float * gMin){
  int i, j;
  for (i = 0; i < POPULATION; i++){
    for (j = 0; j < N; j++){
      float r1 = rand01();
      float r2 = rand01();
      vel[i * N + j] = W * vel[i * N + j] + PHI_P * r1 * (min[i * N + j] - pos[i * N + j]) + PHI_G * r2 * (gMin[j] - pos[i * N + j]);
    }
  }
}

// Updates the position of each particle
void update_pos(float * pos, float * vel){
  int i, j;
  for (i = 0; i < POPULATION; i++){
    for (j = 0; j < N; j++){
      pos[i * N + j] = pos[i * N + j] + vel[i * N + j];
    }
  }
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
  float * pos, * min, * gMin;
  float * evals, * min_evals;
  float * vel;
  pos = (float *) malloc(N * POPULATION * sizeof(float));
  vel = (float *) malloc(N * POPULATION * sizeof(float));
  min = (float *) malloc(N * POPULATION * sizeof(float));
  evals = (float *) malloc(POPULATION * sizeof(float));
  min_evals = (float *) malloc(POPULATION * sizeof(float));
  gMin = (float *) malloc(N * sizeof(float));

/*
  float pos[N * POPULATION];
  float min[N * POPULATION];
  float gMin[N];
  float evals[POPULATION];
  float min_evals[POPULATION];
  float vel[N * POPULATION];
*/

  // Some thins only to save results
  float solutions[N_EXPERIMENTS];
  int stops[N_EXPERIMENTS];

  clock_t start, end;
  double cpu_time_used;

  start = clock();

  int experiment;
  for (experiment = 0; experiment < N_EXPERIMENTS; experiment ++){

    //------------------------- PSO Algorithm ----------------------------------
    // Initialising things
    init_pos(pos);
    init_vel(vel);

    init_min(min);
    init_min_evals(min_evals);
    init_gMin(gMin);

    int iter = 0;
    while (iter < N_ITERS && F_KEY(gMin) > TOLERANCE) {
      eval(pos, evals);
      update_min(pos, min, evals, min_evals);
      update_gMin(min, min_evals, gMin);
      update_vel(pos, vel, min, gMin);
      update_pos(pos, vel);
      iter ++;
    }
    //--------------------------------------------------------------------------

    solutions[experiment] = F_KEY(gMin);
    stops[experiment] = iter;
  }


  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;


  printf("\n--- PSO : Test info ---\n");
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
  printf("[-]  Solution info :\n");
  printf("      Mean : %f\n", sol_mean);
  printf("      Dev  : %f\n", fdev(solutions, fmean(solutions, N_EXPERIMENTS), N_EXPERIMENTS));
  printf("[-]  Max iter info :\n");
  printf("      Mean : %d \n", iter_mean);
  printf("      Dev  : %f\n", idev(stops, imean(stops, N_EXPERIMENTS), N_EXPERIMENTS));
  return 0;
}
