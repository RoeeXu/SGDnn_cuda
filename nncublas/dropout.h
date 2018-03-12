/*****************************************************
* dropout.h
*
*
* dropout layer
*
*
* Created by Roee Xu
*****************************************************/

#include <stdio.h>
#include "define.h"

__global__ void DotBoolKernel(double *a, bool *b, double *c, int size)  
{  
	CUDA_KERNEL_LOOP(i, size){
		if(b[i]) c[i]=a[i];
		else c[i]=0;
	}
}

double * dropout(double * data, bool * mask, int dim){
	double *res;
	cudaMalloc((void**)&res,dim * sizeof(double));
	int BLOCK_NUM = (dim - 1) / THREAD_NUM + 1;
	DotBoolKernel<<< BLOCK_NUM, THREAD_NUM, 0 >>>(data, mask, res, dim);
	return res;
}

double * dropout_bp(double * loss, bool * mask, int dim){
	double *res;
	cudaMalloc((void**)&res,dim * sizeof(double));
	int BLOCK_NUM = (dim - 1) / THREAD_NUM + 1;
	DotBoolKernel<<< BLOCK_NUM, THREAD_NUM, 0 >>>(loss, mask, res, dim);
	return res;
}