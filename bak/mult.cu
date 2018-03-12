#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define max(x,y) (x>y?x:y)
#define min(x,y) (x>y?y:x)

#define THREAD_NUM 256

int BLOCK_NUM=0;

void matgen(double* a, int n, int m)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            a[i * m + j] = (double)rand() / RAND_MAX + 
            (double)rand() / ((long)RAND_MAX * RAND_MAX);
        }
    }
}

__global__ static void MatMultKernel(const double* a, const double* b, double* c, int n, int m, int k)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * THREAD_NUM + tid;
    const int row = idx / k;
    const int column = idx % k;
    if (row < n && column < k)
    {
        double t = 0;
        for (int i = 0; i < m; i++)
        {
            t += a[row * m + i] * b[i * k + column];
        }
        c[idx] = t;
    }
}

void MatMultWithCuda(const double *a, const double *b, double *c, int n, int m, int k){
    double *cuda_a, *cuda_b, *cuda_c;
    cudaMalloc((void**)&cuda_a, sizeof(double)* n * m);
    cudaMalloc((void**)&cuda_b, sizeof(double)* m * k);
    cudaMalloc((void**)&cuda_c, sizeof(double)* n * k);
    cudaMemcpy(cuda_a, a, sizeof(double)* n * m, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, sizeof(double)* m * k, cudaMemcpyHostToDevice);
    BLOCK_NUM = min(n,k) * (max(n,k) + THREAD_NUM - 1) / THREAD_NUM;
    MatMultKernel<<< BLOCK_NUM, THREAD_NUM, 0 >>>(cuda_a , cuda_b , cuda_c , n, m, k);
    cudaMemcpy(c, cuda_c, sizeof(double)* n * k, cudaMemcpyDeviceToHost);
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);
}

int main()
{
    srand(time(NULL));
    double *a, *b, *c;
    int n, m, k;
    scanf("%d%d%d",&n,&m,&k);
    a = (double*)malloc(sizeof(double)* n * m); 
    b = (double*)malloc(sizeof(double)* m * k); 
    c = (double*)malloc(sizeof(double)* n * k); 
    srand(time(NULL));
    matgen(a, n, m);
    matgen(b, m, k);
    MatMultWithCuda(a, b, c, n, m, k);
    for(int i=0;i<n;i++){
    	for(int j=0;j<k;j++){
    		printf("%lf\t",c[i*k+j]);
    	}
    	printf("\n");
    }
    return 0;
}