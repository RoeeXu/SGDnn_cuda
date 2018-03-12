/*****************************************************
* nn_main.cu
*
*
* Neural Network Main Code C++ Cuda Mix
*
*
* Created by Roee Xu
*****************************************************/

#include <iostream>
#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <time.h>
#include "define.h"
#include "read_data.h"
#include "inner_product.h"
#include "softmax_loss.h"
#include "accuracy.h"
#include "random.h"
#include "dropout.h"

using namespace std;

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
	int *d = read_mnist_train_images(s, n, x, y);
	int *l = read_mnist_train_labels(s, n); 
	double *da = new double[n*x*y];
	double *w1 = new double[batch*x*y*ip1od];
	double *b1 = new double[batch*ip1od]();
	double *w2 = new double[batch*ip2od*ip1od];
	double *b2 = new double[batch*ip2od]();
	bool *p1 = new bool[batch*ip1od];
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
	cudaThreadSynchronize();
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
		cudaFree(PX1);
		cudaFree(PU1);
		cudaFree(PD1);
		cudaFree(PU2);
		cudaFree(L);
		cudaFree(U3);
		cudaFree(U2);
		cudaFree(D1);
		cudaFree(U1);
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
	cudaThreadSynchronize();
	cudaFree(xx);
	cudaFree(yy);
	return 0;
}
