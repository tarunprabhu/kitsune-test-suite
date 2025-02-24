#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <stdlib.h>


void random_matrix(float *I, unsigned int rows, unsigned int cols) {
  srand(7);
  using namespace std;
  auto start_time = chrono::steady_clock::now();
  for(unsigned int i = 0 ; i < rows ; i++) {
   for (unsigned int j = 0 ; j < cols ; j++) {
      I[i*cols+j] = rand()/(float)RAND_MAX;
    }
  }

  auto end_time = chrono::steady_clock::now();
  double elapsed_time = chrono::duration<double>(end_time-start_time).count();
  cout << "  random matrix creation time " << elapsed_time << "\n";
  cout << "  initial input data:\n";
  for(unsigned int i = 0; i < 10; i++) {
    cout << "   ";
    for(unsigned int j = 0; j < 10; j++)
      cout << I[i * cols + j] << " ";
    cout << "...\n";
  }
  cout << "   ...\n";
}

void usage(int argc, char **argv)
{
  fprintf(stderr,
        "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <lambda> <no. of iter>\n",
        argv[0]);
  fprintf(stderr, "\t<rows>        - number of rows\n");
  fprintf(stderr, "\t<cols>        - number of cols\n");
  fprintf(stderr, "\t<y1> 	       - y1 value of the speckle\n");
  fprintf(stderr, "\t<y2>          - y2 value of the speckle\n");
  fprintf(stderr, "\t<x1>          - x1 value of the speckle\n");
  fprintf(stderr, "\t<x2>          - x2 value of the speckle\n");
  fprintf(stderr, "\t<lambda>      - lambda (0,1)\n");
  fprintf(stderr, "\t<no. of iter> - number of iterations\n");
  exit(1);
}

__global__ void Loop1(int cols, float *J, int *iN, int *iS, int *jE, int *jW, float *dN, float *dS, float *dE, float *dW, float q0sqr, float *c) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  for(int j = 0; j < cols; j++) {
    int k = i * cols + j;
    float Jc = J[k];
    // directional derivatives
    dN[k] = J[iN[i] * cols + j] - Jc;
    dS[k] = J[iS[i] * cols + j] - Jc;
    dE[k] = J[i * cols + jE[j]] - Jc;
    dW[k] = J[i * cols + jW[j]] - Jc;

    float G2 = (dN[k]*dN[k] + dS[k]*dS[k] +
                dW[k]*dW[k] + dE[k]*dE[k]) / (Jc*Jc);

    float L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

    float num  = (0.5f*G2) - ((1.0f/16.0f)*(L*L)) ;
    float den  = 1 + (.25f*L);
    float qsqr = num/(den*den);

    // diffusion coefficient (equ 33)
    den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
    c[k] = 1.0f / (1.0f+den) ;

    // saturate diffusion coefficient
    if (c[k] < 0)
      c[k] = 0.0;
    else if (c[k] > 1)
      c[k] = 1.0;
  }
}

__global__ void Loop2(int cols, float *J, float *c, int *iS, int *jE, float *dN, float *dS, float *dE, float *dW, float lambda) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  for(int j = 0; j < cols; j++) {
    // current index
    int k = i * cols + j;
    // diffusion coefficient
    float cN = c[k];
    float cS = c[iS[i] * cols + j];
    float cW = c[k];
    float cE = c[i * cols + jE[j]];
    // divergence (equ 58)
    float D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];
    // image update (equ 61)
    J[k] = J[k] + 0.25*lambda*D;
  }
}

int main(int argc, char* argv[])
{
  using namespace std;
  int rows, cols, size_I, size_R, niter;
  float *I, *J, q0sqr, tmp, meanROI,varROI ;
  float Jc;
  int *iN,*iS,*jE,*jW;
  float *dN,*dS,*dW,*dE;
  int r1, r2, c1, c2;
  float *c;
  float lambda;

  if (argc == 9) {
    rows = atoi(argv[1]); //number of rows in the domain
    cols = atoi(argv[2]); //number of cols in the domain
    r1   = atoi(argv[3]); //y1 position of the speckle
    r2   = atoi(argv[4]); //y2 position of the speckle
    c1   = atoi(argv[5]); //x1 position of the speckle
    c2   = atoi(argv[6]); //x2 position of the speckle
    lambda = atof(argv[7]); //Lambda value
    niter = atoi(argv[8]); //number of iterations
  } else if (argc == 1) {
    // run with a default configuration.
    rows = 6400;
    cols = 6400;
    r1 = 0;
    r2 = 127;
    c1 = 0;
    c2 = 127;
    lambda = 0.5;
    niter = 100;
  } else {
    usage(argc, argv);
  }

  if ((rows%16!=0) || (cols%16!=0)){
    fprintf(stderr, "rows and cols must be multiples of 16\n");
    exit(1);
  }

  cout << setprecision(5);
  cout << "\n";
  cout << "---- srad benchmark (cuda) ----\n"
       << "  Row size    : " << rows << ".\n"
       << "  Column size : " << cols << ".\n"
       << "  Iterations  : " << niter << ".\n\n";

  cout << "  Allocating arrays and building random matrix..."
       << std::flush;

  size_I = cols * rows;
  size_R = (r2-r1+1)*(c2-c1+1);

  cudaError_t err = cudaSuccess;
  err = cudaMallocManaged(&I, size_I * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory!\n");
    return 1;
  }
  err = cudaMallocManaged(&J, size_I * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory!\n");
    return 1;
  }
  err = cudaMallocManaged(&c, size_I * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory!\n");
    return 1;
  }
  err = cudaMallocManaged(&iN, rows * sizeof(int));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory!\n");
    return 1;
  }
  err = cudaMallocManaged(&iS, rows * sizeof(int));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory!\n");
    return 1;
  }
  err = cudaMallocManaged(&jW, cols * sizeof(int));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory!\n");
    return 1;
  }
  err = cudaMallocManaged(&jE, cols * sizeof(int));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory!\n");
    return 1;
  }
  err = cudaMallocManaged(&dN, size_I * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory!\n");
    return 1;
  }
  err = cudaMallocManaged(&dS, size_I * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory!\n");
    return 1;
  }
  err = cudaMallocManaged(&dE, size_I * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory!\n");
    return 1;
  }
  err = cudaMallocManaged(&dW, size_I * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory!\n");
    return 1;
  }

  random_matrix(I, rows, cols);


  cout << "  Starting benchmark...\n" << std::flush;
  auto start_time = chrono::steady_clock::now();

  int threadsPerBlock = 256;
  int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

  for(int i = 0; i < rows; i++) {
    iN[i] = i-1;
    iS[i] = i+1;
  }

  for(int j=0; j < cols; j++) {
    jW[j] = j-1;
    jE[j] = j+1;
  }

  iN[0] = 0;
  iS[rows-1] = rows-1;
  jW[0] = 0;
  jE[cols-1] = cols-1;

  for(int k = 0;  k < size_I; k++ )
    J[k] = expf(I[k]) ;

  double loop1_total_time = 0.0;
  double loop2_total_time = 0.0;
  double loop1_max_time = 0.0, loop1_min_time = 1000.0;
  double loop2_max_time = 0.0, loop2_min_time = 1000.0;

  for (int iter=0; iter < niter; iter++) {
    float sum=0, sum2=0;

    for(int i=r1; i <= r2; i++) {
      for(int j = c1; j<=c2; j++) {
        tmp   = J[i * cols + j];
        sum  += tmp ;
        sum2 += tmp*tmp;
      }
    }
    meanROI = sum / size_R;
    varROI  = (sum2 / size_R) - meanROI*meanROI;
    q0sqr   = varROI / (meanROI*meanROI);

    auto loop1_start_time = chrono::steady_clock::now();
    Loop1<<<blocksPerGrid, threadsPerBlock>>>(cols, J, iN, iS, jE, jW, dN, dS, dE, dW, q0sqr, c);
    cudaDeviceSynchronize();
    auto loop1_end_time = chrono::steady_clock::now();
    double etime = chrono::duration<double>
      (loop1_end_time - loop1_start_time).count();
    loop1_total_time += etime;

    if (etime > loop1_max_time)
      loop1_max_time = etime;
    else if (etime < loop1_min_time)
      loop1_min_time = etime;

    auto loop2_start_time = chrono::steady_clock::now();
    Loop2<<<blocksPerGrid, threadsPerBlock>>>(cols, J, c, iS, jE, dN, dS, dE, dW, lambda);
    cudaDeviceSynchronize();
    auto loop2_end_time = chrono::steady_clock::now();
    etime = chrono::duration<double>
      (loop2_end_time - loop2_start_time).count();
    loop2_total_time += etime;
    if (etime > loop2_max_time)
      loop2_max_time = etime;
    else if (etime < loop2_min_time)
      loop2_min_time = etime;
  }
  auto end_time = chrono::steady_clock::now();
  double elapsed_time = chrono::duration<double>
    (end_time - start_time).count();
  cout << "  Avg. loop 1 time: " << loop1_total_time / niter << "\n"
       << "       loop 1 min : " << loop1_min_time << "\n"
       << "       loop 1 max : " << loop1_max_time << "\n"
       << "  Avg. loop 2 time: " << loop2_total_time / niter << "\n"
       << "       loop 2 min : " << loop2_min_time << "\n"
       << "       loop 2 max : " << loop2_max_time << "\n";
  cout << "  Running time: " << elapsed_time << " seconds.\n"
       << "*** " << elapsed_time << ", " << elapsed_time << "\n"
       << "----\n\n";

  FILE *fp = fopen("srad-cuda-output.dat", "wb");
  if (fp != NULL) {
    fwrite((void*)J, sizeof(float), size_I, fp);
    fclose(fp);
  }
  return 0;
}
