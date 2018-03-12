/*****************************************************
* inner_product.h
*
*
* inner product layer
*
*
* Created by Roee Xu
*****************************************************/

#include <cstdio>

double * inner_product(double * data, double * weight, double * bias, int batch, int indim, int outdim){
	double * res = new double[batch * outdim]();
	for(int i = 0;i < batch;i++)
		for(int j = 0;j < outdim;j++){
			for(int k = 0;k < indim;k++)
				res[i * outdim + j] += data[i * indim + k] * weight[i * indim * outdim + j * indim + k];
			res[i * outdim + j] += bias[i * outdim + j];
		}
	return res;
}

double * inner_product_bp(double * u, double * x, double * w, double * b, 
						  double * loss, double lr, int batch, int indim, int outdim){
	double * res = new double[batch * indim]();
	for(int i = 0;i < batch;i++)
		for(int j = 0;j < outdim;j++){
			for(int k = 0;k < indim;k++){
				res[i * indim + k] += w[i * indim * outdim + j * indim + k] * loss[i * outdim + j];
				w[i * indim * outdim + j * indim + k] -= lr * (loss[i * outdim + j] * x[i * indim + k]);
			}
			b[i * outdim + j] -= lr * loss[i * outdim + j];
		}
	return res;
}