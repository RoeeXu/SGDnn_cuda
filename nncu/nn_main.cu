/*****************************************************
* nn_main.cu
*
*
* Neural Network Main Code C++ Cuda Mix
*
*
* Created by Roee Xu
*****************************************************/

#include <stdio.h>
#include <string>
#include <math.h>
#include <cuda_runtime.h>
#include <iostream>
#include "read_data.hpp"

using namespace std;

#define max(x,y) (x>y?x:y)
#define min(x,y) (x>y?y:x)

#define CUDA_KERNEL_LOOP(i, n) \
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
		i < (n); \
		i += blockDim.x * gridDim.x)

#define THREAD_NUM 1024

__global__ void AddKernel(double *a, double *b, double *c, int size)  
{  
    CUDA_KERNEL_LOOP(i, size)
    	c[i] = a[i] + b[i];
}

__global__ void SubKernel(double *a, double *b, double *c, int size)  
{  
    CUDA_KERNEL_LOOP(i, size)
    	c[i] = a[i] - b[i];
}

__global__ void DotBoolKernel(double *a, bool *b, double *c, int size)  
{  
    CUDA_KERNEL_LOOP(i, size){
    	if(b[i]) c[i] = a[i];
    	else c[i]=0;
    }
}

__global__ void DotKernel(double *a, double *b, double *c, int size)  
{  
    CUDA_KERNEL_LOOP(i, size)
    	c[i] = a[i] * b[i];
}

__global__ void MultKernel(double *a, double *b, int size)  
{  
    CUDA_KERNEL_LOOP(i, size)
    	a[i] *= b[0];
}

__global__ void UpdateKernel(double *a, double *b, double *c, int size)
{  
    CUDA_KERNEL_LOOP(i, size)
    	a[i] -= (c[0] * b[i]);
}

__global__ void MatMultKernel(double *a, double *b, double *c, int n, int m, int k, int batch)
{
    CUDA_KERNEL_LOOP(i, n*k*batch){
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
}

__global__ void ExpKernel(double * a, double * b, int size)
{
	CUDA_KERNEL_LOOP(i, size)
    	b[i]=exp(a[i]);
}

__global__ void SumKernel(double * a, double * b, int batch, int dim)
{
	CUDA_KERNEL_LOOP(i, batch){
    	b[i]=0;
    	for(int j=0;j<dim;j++){
    		b[i] += a[i*dim+j];
    	}
    }
}

__global__ void DivKernel(double * a, double * b, double * c, int batch, int dim)
{
	CUDA_KERNEL_LOOP(i, batch*dim){
		int j=i/dim;
		c[i]=a[i]/b[j];
	}
}

__global__ void LossNeglogKernel(double * a, int * b, double * c, int batch, int dim)
{
	CUDA_KERNEL_LOOP(i, batch)
    	c[i]=-log(a[i * dim + b[i]]);
}

__global__ void InitKernel(double * a, double value, int size)
{
	CUDA_KERNEL_LOOP(i, size)
    	a[i]=value;
}

__global__ void SoftmaxBpKernel(double * u, int * label, double * res, int batch, int dim)
{
	CUDA_KERNEL_LOOP(i, dim*batch){
		int x = i/dim;
	    int y = i%dim;
    	if(y==label[x])
    		res[x*dim+y]=u[x*dim+y]-1;
    	else
    		res[x*dim+y]=u[x*dim+y];
	}
}

__global__ void AddMultKernel(double *a, double *b, double *c, double *d, int n, int m, int k, int batch)
{
	CUDA_KERNEL_LOOP(i, n*k*batch){
		int x = i/(n*k);
	    int y = (i-x*n*k)/k;
	    int z = i%k;
	    if(x<batch){
	    	double t=c[i];
	    	for(int j=0;j<m;j++)
	    		t += a[x*n*m+y*m+j] * b[x*m*k+j*k+z];
	    	d[i] = t;
	    }
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
	int BLOCK_NUM = (batch*outdim + THREAD_NUM - 1) / THREAD_NUM;
	AddMultKernel<<< BLOCK_NUM, THREAD_NUM, 0 >>>(weight, data, bias, res, outdim, indim, 1, batch);
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
	int BLOCK_NUM = (batch*indim*outdim - 1) / THREAD_NUM + 1;
	UpdateKernel<<< BLOCK_NUM, THREAD_NUM, 0 >>>(w,dw,ll,batch*indim*outdim);
	BLOCK_NUM = (batch*outdim - 1) / THREAD_NUM + 1;
	UpdateKernel<<< BLOCK_NUM, THREAD_NUM, 0 >>>(b,loss,ll,batch*outdim);
	cudaFree(dw);
	cudaFree(ll);
	return res;
}

double * dropout(double * data, bool * mask, int dim){
	double * res;
	cudaMalloc((void**)&res, dim * sizeof(double));
	int BLOCK_NUM = (dim - 1) / THREAD_NUM + 1;
	DotBoolKernel<<< BLOCK_NUM, THREAD_NUM, 0 >>>(data, mask, res, dim);
	return res;
}

double * dropout_bp(double * loss, bool * mask, int dim){
	double * res;
	cudaMalloc((void**)&res, dim * sizeof(double));
	int BLOCK_NUM = (dim - 1) / THREAD_NUM + 1;
	DotBoolKernel<<< BLOCK_NUM, THREAD_NUM, 0 >>>(loss, mask, res, dim);
	return res;
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

int compute_correct_num(double * data, int * label, int batch, int dim){
	int res=0;
	for(int i=0;i<batch;i++){
		int l=-1;
		double max=0;
		for(int j=0;j<dim;j++){
			if(data[i*dim+j]>max){
				max=data[i*dim+j];
				l=j;
			}
		}
		if(l==label[i]) res++;
	}
	return res;
}

int main(int argc,char * argv[]){
	int batch=100;
	int ip1od=200;
	int ip2od=10;
	double lr=0.01;
	double gama=0.9999;
	int iter = 10000;
	double dropratio=0.5;
	int n,x,y;
	int pos=0;
	int nowiter=0;
	string s=argv[1];
	srand(time(NULL));
	int * d = read_mnist_train_images(s, n, x, y);
	int * l = read_mnist_train_labels(s, n);
	double * da = new double[n*x*y];
	double * w1 = new double[batch*x*y*ip1od];
	double * b1 = new double[batch*ip1od]();
	double * w2 = new double[batch*ip2od*ip1od];
	double * b2 = new double[batch*ip2od]();
	bool * p1 = new bool[batch*ip1od];
	for(int i=0;i<n*x*y;i++)
		da[i]=d[i]/255.0;
	for(int i=0;i<batch*x*y*ip1od;i++)
		w1[i]=gaussrand(0,0.01);
	for(int i=0;i<batch*ip2od*ip1od;i++)
		w2[i]=gaussrand(0,0.01);
	for(int i=0;i<batch*ip1od;i++)
		p1[i]=boolrand(dropratio);
	double *data;
	int *label;
	double *W1;
	double *B1;
	double *W2;
	double *B2;
	bool *P1;
	cudaMalloc((void**)&data, n*x*y * sizeof(double));
	cudaMalloc((void**)&label, n * sizeof(int));
	cudaMalloc((void**)&W1, batch*x*y*ip1od * sizeof(double));
	cudaMalloc((void**)&B1, batch*ip1od * sizeof(double));
	cudaMalloc((void**)&W2, batch*ip2od*ip1od * sizeof(double));
	cudaMalloc((void**)&B2, batch*ip2od * sizeof(double));
	cudaMalloc((void**)&P1, batch*ip1od * sizeof(bool));
	cudaMemcpy(data, da, n*x*y * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(label, l, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(W1, w1, batch*x*y*ip1od * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(B1, b1, batch*ip1od * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(W2, w2, batch*ip2od*ip1od * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(B2, b2, batch*ip2od * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(P1, p1, batch*ip1od * sizeof(bool), cudaMemcpyHostToDevice);
	delete []l;
	delete []d;
	delete []da;
	delete []w1;
	delete []b1;
	delete []w2;
	delete []b2;
	delete []p1;
	while(nowiter++<iter){
		double *now_d=data+pos*x*y;
		int *now_l=label+pos;
		double * U1 = inner_product(now_d,W1,B1,batch,x*y,ip1od);
		double * D1 = dropout(U1,P1,batch*ip1od);
		double * U2 = inner_product(D1,W2,B2,batch,ip1od,ip2od);
		double * U3 = softmax(U2,batch,ip2od);
		double * L = loss(U3,now_l,batch,ip2od);
		double * ll = new double[1];
		cudaMemcpy(ll, L, sizeof(double), cudaMemcpyDeviceToHost);
		cout<<"Iteration: "<<nowiter<<"\tlr: "<<lr<<"\tTrain loss: "<<ll[0]<<endl;
		double * PU2 = softmax_bp(U3, now_l, batch, ip2od);
		double * PD1 = inner_product_bp(U2, D1, W2, B2, PU2, lr, batch, ip1od, ip2od);
		double * PU1 = dropout_bp(PD1,P1,batch*ip1od);
		double * PX1 = inner_product_bp(U1, now_d, W1, B1, PU1, lr, batch, x*y, ip1od);
		lr*=gama;
		pos+=batch;
		pos%=n;
		cudaFree(PU2);
		cudaFree(PD1);
		cudaFree(PU1);
		cudaFree(PX1);
		cudaFree(U1);
		cudaFree(D1);
		cudaFree(U2);
		cudaFree(U3);
		cudaFree(L);
	}
	cudaFree(data);
	cudaFree(label);
	int * e = read_mnist_test_images(s, n, x, y);
	int * m = read_mnist_test_labels(s, n);
	double * ea = new double[n*x*y];
	for(int i=0;i<n*x*y;i++)
		ea[i]=e[i]/255.0;
	double *xx;
	int *yy;
	cudaMalloc((void**)&xx, n*x*y * sizeof(double));
	cudaMalloc((void**)&yy, n * sizeof(int));
	cudaMemcpy(xx, ea, n*x*y * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(yy, m, n * sizeof(int), cudaMemcpyHostToDevice);
	delete []e;
	delete []m;
	delete []ea;
	int times=n/batch;
	pos=0;
	int num=0;
	while(times--){
		double *now_d=xx+pos*x*y;
		int *now_l=yy+pos;
		double * U1 = inner_product(now_d,W1,B1,batch,x*y,ip1od);
		double * D1 = dropout(U1,P1,batch*ip1od);
		double * U2 = inner_product(D1,W2,B2,batch,ip1od,ip2od);
		double * U3 = softmax(U2,batch,ip2od);
		double * ut = new double[batch*ip2od];
		int * lt = new int[batch];
		cudaMemcpy(ut, U3, batch*ip2od*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(lt, now_l, batch*sizeof(int), cudaMemcpyDeviceToHost);
		num += compute_correct_num(ut,lt,batch,ip2od);
		pos+=batch;
		pos%=n;
		delete []ut;
		delete []lt;
		cudaFree(U1);
		cudaFree(D1);
		cudaFree(U2);
		cudaFree(U3);
	}
	cout<<"Test accuracy: "<<num/(n+0.0)<<endl;
	cudaFree(xx);
	cudaFree(yy);
	return 0;
}
