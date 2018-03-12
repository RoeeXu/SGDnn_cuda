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
#include <cblas.h>
#include <string.h>

double * inner_product(double * data, double * weight, double * bias, int batch, int indim, int outdim){
	double *res = new double[batch * outdim];
	memcpy(res, bias, batch*outdim*sizeof(double));
	for(int i=0;i<batch;i++)
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
			outdim, 1, indim, 1.0, weight+i*indim*outdim, indim, 
			data+i*indim, 1, 1.0, res+i*outdim, 1);
	return res;
}

double * inner_product_bp(double * u, double * x, double * w, double * b, 
						  double * loss, double lr, int batch, int indim, int outdim){
	double * res = new double[batch * indim]();
	for(int i=0;i<batch;i++){
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
			1, indim, outdim, 1.0, loss+i*outdim, outdim, 
			w+i*indim*outdim, indim, 0.0, res+i*indim, indim);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
			outdim, indim, 1, -lr, loss+i*outdim, 1, 
			x+i*indim, indim, 1.0, w+i*indim*outdim, indim);
		cblas_daxpy(outdim, -lr, loss+i*outdim, 1, b+i*outdim, 1);
	}
	return res;
}