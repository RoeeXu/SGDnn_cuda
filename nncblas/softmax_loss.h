/*****************************************************
* softmax_loss.h
*
*
* softmax loss layer
*
*
* Created by Roee Xu
*****************************************************/

#include <cstdio>
#include <cmath>

#define max(x,y) (x>y?x:y)

double * softmax(double * data, int batch, int dim){
	double * res = new double[batch * dim];
	double sum;
	for(int i = 0;i < batch;i++)
	{
		sum = 0;
		for(int j = 0;j < dim;j++)
		{
			res[i * dim + j] = exp(data[i * dim + j]);
			sum += res[i * dim + j];
		}
		for(int j = 0;j < dim;j++)
			res[i * dim + j] /= sum;
	}
	return res;
}

double loss(double * data, int * label, int batch, int dim){
	double loss = 0;
	for(int i = 0;i < batch;i++)
		loss += -log(data[i * dim + label[i]]);
	loss /= batch;
	return loss;
}

double * softmax_bp(double * u, int * label, int batch, int dim){
	double * res = new double[batch * dim];
	for(int i=0;i<batch;i++){
		for(int j=0;j<dim;j++){
			if(j==label[i])
				res[i*dim+j]=u[i*dim+j]-1;
			else
				res[i*dim+j]=u[i*dim+j];
		}
	}
	return res;
}