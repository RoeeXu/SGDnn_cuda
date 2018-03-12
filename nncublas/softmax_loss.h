/*****************************************************
* softmax_loss.h
*
*
* softmax loss layer
*
*
* Created by Roee Xu
*****************************************************/

#include <stdio.h>
#include "define.h"

#define max(x,y) (x>y?x:y)

__global__ void ExpKernel(double * a, double * b, int size)
{
	int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * THREAD_NUM + tid;
    if(i<size) b[i]=exp(a[i]);
}

__global__ void SumKernel(double * a, double * b, int batch, int dim)
{
	int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * THREAD_NUM + tid;
    if(i<batch){
    	b[i]=0;
    	for(int j=0;j<dim;j++){
    		b[i]+=a[i*dim+j];
    	}
    }
}

__global__ void DivKernel(double * a, double * b, double * c, int batch, int dim)
{
	int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * THREAD_NUM + tid;
    int bb = i/dim;
    if(bb<batch&&i<batch*dim){
    	c[i]=a[i]/b[bb];
    }
}

__global__ void LossNeglogKernel(double * a, int * b, double * c, int batch, int dim)
{
	int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * THREAD_NUM + tid;
    if(i<batch){
    	c[i]=-log(a[i * dim + b[i]]);
    }
}

__global__ void SoftmaxBpKernel(double * u, int * label, double * res, int batch, int dim)
{
	int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * THREAD_NUM + tid;
    int x = i/dim;
    int y = i%dim;
    if(x<batch&&y<dim){
    	if(y==label[x])
    		res[x*dim+y]=u[x*dim+y]-1;
    	else
    		res[x*dim+y]=u[x*dim+y];
    }
}

double * softmax(double * data, int batch, int dim){
	double * res,*t;
	cudaMalloc((void**)&res, batch * dim * sizeof(double));
	cudaMalloc((void**)&t, batch * dim * sizeof(double));
	double * sum;
	cudaMalloc((void**)&sum, batch * sizeof(double));
	int BLOCK_NUM = (batch*dim - 1) / THREAD_NUM + 1;
	ExpKernel<<< BLOCK_NUM, THREAD_NUM, 0 >>>(data, t, batch*dim);
	BLOCK_NUM = (batch - 1) / THREAD_NUM + 1;
	SumKernel<<< BLOCK_NUM, THREAD_NUM, 0 >>>(t, sum, batch, dim);
	BLOCK_NUM = (batch*dim - 1) / THREAD_NUM + 1;
	DivKernel<<< BLOCK_NUM, THREAD_NUM, 0 >>>(t, sum, res, batch, dim);
	cudaFree(t);
	cudaFree(sum);
	return res;
}

double * loss(double * data, int * label, int batch, int dim){
	double *t,*loss,*v,*u;
	cudaMalloc((void**)&t, batch * sizeof(double));
	cudaMalloc((void**)&loss, sizeof(double));
	cudaMalloc((void**)&v, sizeof(double));
	cudaMalloc((void**)&u, sizeof(double));
	int BLOCK_NUM = (batch - 1) / THREAD_NUM + 1;
	LossNeglogKernel<<< BLOCK_NUM, THREAD_NUM, 0 >>>(data, label, t, batch, dim);
	SumKernel<<< 1, 1, 0 >>>(t, u, 1, batch); 
	InitKernel<<< 1, 1, 0 >>>(v, (double)batch, 1); 
	DivKernel<<< 1, 1, 0 >>>(u, v, loss, 1, 1);
	cudaFree(t);
	cudaFree(u); 
	cudaFree(v);
	return loss;
}

double * softmax_bp(double * u, int * label, int batch, int dim){
	double * res = new double[batch * dim];
	cudaMalloc((void**)&res, batch * dim * sizeof(double));
	int BLOCK_NUM = (batch*dim - 1) / THREAD_NUM + 1;
	SoftmaxBpKernel<<< BLOCK_NUM, THREAD_NUM, 0 >>>(u, label, res, batch, dim);
	return res;
}