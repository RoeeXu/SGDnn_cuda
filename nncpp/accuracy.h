/*****************************************************
* accuracy.h
*
*
* compute correct number
*
*
* Created by Roee Xu
*****************************************************/

#include <cstdio>

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