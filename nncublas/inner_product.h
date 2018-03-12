/*****************************************************
* inner_product.h
*
*
* inner product layer
*
*
* Created by Roee Xu
*****************************************************/

#include <stdio.h>
#include "define.h"

__global__ void MatMultKernel(double* a, double* b, double* c, int n, int m, int k, int batch)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * THREAD_NUM + tid;
    int x = i/(n*k);
    int y = (i-x*n*k)/k;
    int z = i%k;
    if(x<batch){
    	double t=0;
    	for(int j=0;j<m;j++)
    		t += a[x*n*m+y*m+j] * b[x*m*k+j*k+z];
    	c[i] = t;
    }
}

__global__ void AddKernel(double *a, double *b, double *c, int size)  
{  
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * THREAD_NUM + tid;
    if(i<size) c[i] = a[i] + b[i];
}

__global__ void SubKernel(double *a, double *b, double *c, int size)  
{  
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * THREAD_NUM + tid;
    if(i<size) c[i] = a[i] - b[i];
}

__global__ void MultKernel(double *a, double *b, int size)  
{  
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * THREAD_NUM + tid;
    if(i<size) a[i] *= b[0];
}

__global__ void InitKernel(double * a, double value, int size)
{
	int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * THREAD_NUM + tid;
    if(i<size){
    	a[i]=value;
    }
}

void AddWithCuda(double *a, double *b, double *c, int size)
{
	int BLOCK_NUM = (size - 1) / THREAD_NUM + 1;
	AddKernel<<< BLOCK_NUM, THREAD_NUM, 0 >>>(a, b, c, size);
}

void MatMultWithCuda(double *a, double *b, double *c, int n, int m, int k, int batch){
	int BLOCK_NUM = (batch*n*k + THREAD_NUM - 1) / THREAD_NUM;
	MatMultKernel<<< BLOCK_NUM, THREAD_NUM, 0 >>>(a , b , c , n, m, k, batch);
}

double * inner_product(double * data, double * weight, double * bias, int batch, int indim, int outdim){
	double * res;
	cudaMalloc((void**)&res, batch * outdim * sizeof(double));
	MatMultWithCuda(weight,data,res,outdim,indim,1,batch);
	AddWithCuda(res,bias,res,outdim*batch);
	return res;
}

double * inner_product_bp(double * u, double * x, double * w, double * b,
						  double * loss, double lr, int batch, int indim, int outdim){
	double *res,*dw,*ll;
	cudaMalloc((void**)&res, batch * indim * sizeof(double));
	cudaMalloc((void**)&dw, batch * indim * outdim * sizeof(double));
	cudaMalloc((void**)&ll, sizeof(double));
	InitKernel<<< 1, 1, 0 >>>(ll, lr, 1);
	MatMultWithCuda(loss,w,res,1,outdim,indim,batch);
	MatMultWithCuda(loss,x,dw,outdim,1,indim,batch);
	for(int i=0;i<batch;i++){
		int BLOCK_NUM = (indim*outdim - 1) / THREAD_NUM + 1;
		MultKernel<<< BLOCK_NUM, THREAD_NUM, 0 >>>(dw+i*indim*outdim, ll, indim*outdim);
		SubKernel<<< BLOCK_NUM, THREAD_NUM, 0 >>>(w+i*indim*outdim, dw+i*indim*outdim, w+i*indim*outdim, indim*outdim);
		BLOCK_NUM = (outdim - 1) / THREAD_NUM + 1;
		MultKernel<<< BLOCK_NUM, THREAD_NUM, 0 >>>(loss+i*outdim,ll,outdim);
		SubKernel<<< BLOCK_NUM, THREAD_NUM, 0 >>>(b+i*outdim, loss+i*outdim, b+i*outdim, outdim);
	}
	cudaFree(dw);
	cudaFree(ll);
	return res;
}

// double * inner_product(double * data, double * weight, double * bias, int batch, int indim, int outdim){
// 	double *res;
// 	cudaMalloc((void**)&res,batch * outdim * sizeof(double));
// 	cudaMemcpy(res, bias, batch * outdim * sizeof(double), cudaMemcpyDeviceToDevice);
// 	cublasHandle_t handle;
//     cublasCreate(&handle);
//     double a=1,b=1;
// 	for(int i=0;i<batch;i++)
// 		cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, 
// 			outdim, 1, indim, &a, weight+i*indim*outdim, outdim, 
// 			data+i*indim, indim, &b, res+i*outdim, outdim);
// 	cublasDestroy(handle);
// 	return res;
// }

// double * inner_product_bp(double * u, double * x, double * w, double * b, 
// 						  double * loss, double lr, int batch, int indim, int outdim){
// 	double * res;
// 	cudaMalloc((void**)&res,batch * indim * sizeof(double));
// 	cublasHandle_t handle;
//     cublasCreate(&handle);
//     double a=1,bb=0,c=-lr;
// 	for(int i=0;i<batch;i++){
// 		cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, 
// 			1, indim, outdim, &a, loss+i*outdim, 1, 
// 			w+i*indim*outdim, outdim, &bb, res+i*indim, 1);
// 		cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, 
// 			outdim, indim, 1, &c, loss+i*outdim, outdim, 
// 			x+i*indim, 1, &a, w+i*indim*outdim, outdim);
// 		cublasDaxpy(handle,outdim, &c, loss+i*outdim, 1, b+i*outdim, 1);
// 	}
// 	cublasDestroy(handle);
// 	return res;
// }