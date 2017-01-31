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
#define UB 2 // Upper bound
#define LB -2 // Lower bound

// Control parameters
#define LIMIT 100

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

// Prints an array of ints
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

// Generates a random integer beteen two given floats
int ibounded_rand(float low, float up){
  return (int) bounded_rand(low, up);
}

// Generates a random normal number with mean 0 and deviation 1
double gaussrand(){
	static double V1, V2, S;
	static int phase = 0;
	double X;

	if(phase == 0) {
		do {
			double U1 = (double)rand() / RAND_MAX;
			double U2 = (double)rand() / RAND_MAX;

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
			} while(S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	} else
		X = V2 * sqrt(-2 * log(S) / S);

	phase = 1 - phase;

	return X;
}

// Fills an array with a given integers
void ifill(int * arr, int lenght, int number){
  int i;
  for (i = 0; i < lenght; i ++){
    arr[i] = number;
  }
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

// Función a optimizar : Drop-wave
float WAV(float * x){
  float output, sum;
  int i, j;

  sum = 0;
  for (i = 0; i < N; i ++){sum += pow(x[i],2);}
  output = - (1 + cos(12 * sqrt(sum)))/(0.5 * sum + 2);

  return output;
}

// Inits population
void init_bees(float * employees){
  int i, j;
  for (i = 0; i < POPULATION; i ++){
    for (j = 0; j < N; j++){
      employees[i * N + j] = bounded_rand(LB, UB);
    }
  }
}

void init_counter(int * counter){
  ifill(counter, POPULATION, 0);
}

// Seeks around all souces
void seek_around(float * bees, float * new_sources){
  int i, j;
  int k; // other random bee
  float phi;
  for (i = 0; i < POPULATION; i++){
    k = ibounded_rand(0,N);
    for (j = 0; j < N; j++){
      phi = gaussrand();
      new_sources[i * N + j] = bees[i * N + j] + phi * (bees[i * N + j] - bees[k * N + j]);
    }
  }
}

// Evaluates the function fitness in each bees
void fitness(float * bees, float * fit){
  int i;
  for (i = 0; i < POPULATION; i ++){

    if (F_KEY(&bees[i * N]) < 0){
      fit[i] = 1 - F_KEY(&bees[i * N]);
    } else {
      fit[i] = 1/(1 + F_KEY(&bees[i * N]));
    }
  }
}


// Keeps the best source_info with counter
void keep_best_sources_c(float * sources, float * new_sources, float * fit, float * new_fit, int * counter){
  int i, j;
  for (i = 0; i < POPULATION; i ++){
    if (new_fit[i] > fit[i]){
      // Update the fitness
      fit[i] = new_fit[i];
      // Copy the new position
      for (j = 0; j < N; j++){
        sources[i * N + j] = new_sources[i * N + j];
      }
      // Reset the counter
      counter[i] = 0;
    } else {
      counter[i] += 1;
    }
  }
}

// Keeps the best source_info
void keep_best_sources(float * sources, float * new_sources, float * fit, float * new_fit){
  int i, j;
  for (i = 0; i < POPULATION; i ++){
    if (new_fit[i] > fit[i]){
      // Update the fitness
      fit[i] = new_fit[i];
      // Copy the new position
      for (j = 0; j < N; j++){
        sources[i * N + j] = new_sources[i * N + j];
      }
    }
  }
}

// Select an index given an array of fitness/probs
int roulette(float * fit){
  float accum[POPULATION];
  int i;
  accum[0] = fit[0];
  // Calculate the scan
  for (i = 1; i < POPULATION; i++){
    accum[i] =  accum[i - 1] + fit[i];
  }
  // Normalize
  for (i = 0; i < POPULATION; i++){
    accum[i] = accum[i] / accum[POPULATION - 1];
  }
  // Get the random and choose_best_members
  float r = rand01();
  //printf("\nrand : %f\n", r);
  i = 0;
  while (r > accum[i]){
    i++;
  };
  //printf("sol : %d\n", i);
  return i;
}


// Set the destinations
void set_destinations(float * efit, int * destinations, int * occupancy){
  int bee;
  int destination;

  for (bee = 0; bee < POPULATION; bee++){
    destination = roulette(efit);
    destinations[bee] = destination;
    occupancy[destination] += 1;
  }
}

// Sends the onlookers
void send_onlookers(float * onlookers, float * employees, int * destinations){
  int i, j;
  float r;
  int k;
  for (i = 0; i < POPULATION; i++){
    r = gaussrand();
    k = ibounded_rand(0,POPULATION);
    for (j = 0; j < N; j ++){
      onlookers[i * N + j] = employees[destinations[i] * N + j] + r * (employees[destinations[i] * N + j] - employees[k * N + j]);
    }
  }
}

// Updates the employees after onlookers have been send_scouts
void update_employees(float * employees, float * onlookers, float * efit, float * ofit, int * destinations){
  int onlooker, employee, i;

  for (onlooker = 0; onlooker < POPULATION; onlooker++){
    // Gettin the index of the employee associated
    employee = destinations[onlooker];
    if (ofit[onlooker] > efit[employee]){
      efit[employee] = ofit[onlooker];
      for (i = 0; i < N; i ++){
        employees[employee * N + i] = onlookers[onlooker * N + i];
      }
    }
  }
}

// Send Scouts
void send_scouts(float * employees, int * counter){
  int i, j;

  for (i = 0; i < POPULATION; i ++){
    if (counter[i] < LIMIT){
      for (j = 0; j < N; j ++){
        employees[i * N + j] = bounded_rand(LB, UB);
      }
      counter[i] = 0;
    }
  }
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

// Returns the position of the min in an array
int min_index(float * arr, int n){
  int index = 0;
  int i;
  for (i = 0; i < n; i++){
    if (arr[i] < arr[index]){
      index = i;
    }
  }
  return index;
}

// Fill the evals array
void fill_evals(float * pos, float * evals){
  int i;
  for (i = 0; i < POPULATION; i ++){
    evals[i] = F_KEY(&pos[i * N]);
  }
}

int main(void){
  srand (time(NULL));

  // Init the stuff
  float * employees, * onlookers;
  float * new_sources;
  int * destinations;
  float * efit, * ofit, * new_fit, * dest_fit;
  int * counter, * occupancy;
  float gMin_val = 99999;;
  float evals[POPULATION];

  employees = (float *) malloc(N * POPULATION * sizeof(float));
  onlookers = (float *) malloc(N * POPULATION * sizeof(float));



  destinations = (int *) malloc(POPULATION * sizeof(int));
  new_sources = (float *) malloc(N * POPULATION * sizeof(float));
  efit = (float *) malloc(POPULATION * sizeof(float));
  ofit = (float *) malloc(POPULATION * sizeof(float));
  dest_fit = (float *) malloc(POPULATION * sizeof(float));
  new_fit = (float *) malloc(POPULATION * sizeof(float));

  counter = (int *) malloc(POPULATION * sizeof(int));
  occupancy = (int *) malloc(POPULATION * sizeof(int));

  // Some thins only to save results
  float solutions[N_EXPERIMENTS];
  int stops[N_EXPERIMENTS];

  clock_t start, end;
  double cpu_time_used;

  start = clock();

  int experiment;
  for (experiment = 0; experiment < N_EXPERIMENTS; experiment ++){

      //------------------------- ABC Algorithm ----------------------------------

      init_counter(counter);
      init_bees(employees);
      gMin_val = 99999.9;
      fitness(onlookers, ofit);

      int iter = 0;
      while (iter < N_ITERS && gMin_val > TOLERANCE) {

        // Employees --------------------------------
        fitness(employees, efit);
        seek_around(employees, new_sources);
        fitness(new_sources, new_fit);
        keep_best_sources_c(employees, new_sources, efit, new_fit, counter);

        // Onlookers ------------------------------------
        ifill(occupancy, POPULATION, 0);
        set_destinations(efit, destinations, occupancy);
        send_onlookers(onlookers, employees, destinations);
        fitness(onlookers, ofit);
        update_employees(employees, onlookers, efit, ofit, destinations);

        // Scouts --------------------------------------
        fill_evals(employees, evals);
        gMin_val = fmmin(evals, POPULATION);
        iter ++;
      }

    //--------------------------------------------------------------------------

    solutions[experiment] = gMin_val;
    stops[experiment] = iter;

  }

  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;


  printf("\n--- ABC : Test info ---\n");
  printf("[-] Population       : %d\n", POPULATION);
  printf("[-] Variables        : %d\n", N);
  printf("[-] Num. max iters   : %d\n", N_ITERS);
  printf("[-] Num. experiments : %d\n", N_EXPERIMENTS);
  printf("[-] Required time        : %f\n", cpu_time_used);
  printf("[-] Required time mean   : %f\n", cpu_time_used/N_EXPERIMENTS);

  float sol_mean = fmean(solutions, N_EXPERIMENTS);
  int iter_mean = imean(stops, N_EXPERIMENTS);

  printf("\n--- Results ---\n");
  printf("[-] Best solution : %f\n", gMin_val);
  printf("[-] Solution info :\n");
  printf("[-]    Mean : %f\n", sol_mean);
  printf("[-]    Dev  : %f\n", fdev(solutions, fmean(solutions, N_EXPERIMENTS), N_EXPERIMENTS));
  printf("[-] Max iter info :\n");
  printf("[-]    Mean : %d \n", iter_mean);
  printf("[-]    Dev  : %f \n\n", idev(stops, imean(stops, N_EXPERIMENTS), N_EXPERIMENTS));

  return 0;
}
