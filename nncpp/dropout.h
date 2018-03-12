/*****************************************************
* dropout.h
*
*
* dropout layer
*
*
* Created by Roee Xu
*****************************************************/

#include <cstdio>

double * dropout(double * data, int * mask, int dim){
	double * res = new double[dim];
	for(int i=0;i<dim;i++){
		if(mask[i]) res[i]=data[i];
		else res[i]=0;
	}
	return res;
}

double * dropout_bp(double * loss, int * mask, int dim){
	double * res = new double[dim];
	for(int i=0;i<dim;i++){
		if(mask[i]) res[i]=loss[i];
		else res[i]=0;
	}
	return res;
}