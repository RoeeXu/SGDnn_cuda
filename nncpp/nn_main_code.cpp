/*****************************************************
* nn_main_code.cpp
*
*
* Neural Network Main Code C++
*
*
* Created by Roee Xu
*****************************************************/

#include "read_mnist_data.h"
#include "inner_product.h"
#include "softmax_loss.h"
#include "accuracy.h"
#include "random.h"
#include "dropout.h"
#include <iostream>

typedef unsigned char uint8_t;

int main(int argc,char * argv[]){
	int batch=100;
	int ip1od=200;
	int ip2od=10;
	double lr=0.01;
	double gama = 0.9999;
	int iter = 10000;
	double dropratio=0.5;


	int n,x,y;
	int pos=0;
	int nowiter=0;

	string s=argv[1];

	srand(time(NULL));

	uint8_t * d = read_mnist_train_images(s, n, x, y);
	uint8_t * l = read_mnist_train_labels(s, n); 

	double * w1 = new double[batch*x*y*ip1od];
	double * b1 = new double[batch*ip1od]();
	double * w2 = new double[batch*ip2od*ip1od];
	double * b2 = new double[batch*ip2od]();
	int * p1 = new int[batch*ip1od];
	for(int i=0;i<batch*x*y*ip1od;i++)
		w1[i]=gaussrand(0,0.01);
	for(int i=0;i<batch*ip2od*ip1od;i++)
		w2[i]=gaussrand(0,0.01);
	for(int i=0;i<batch*ip1od;i++)
		p1[i]=boolrand(dropratio);
	while(nowiter++<iter){
		double * data = new double[batch*x*y];
		int * label = new int[batch];
		for(int i=0;i<batch*x*y;i++)
			data[i]=(int)d[i+pos*x*y]/255.0;
		for(int i=0;i<batch;i++)
			label[i]=(int)l[i+pos];
		double * u1 = inner_product(data,w1,b1,batch,x*y,ip1od);
		double * d1 = dropout(u1,p1,batch*ip1od);
		double * u2 = inner_product(d1,w2,b2,batch,ip1od,ip2od);
		double * u3 = softmax(u2,batch,ip2od);
		double L = loss(u3,label,batch,ip2od);
		cout<<"Iteration: "<<nowiter<<"\tlr: "<<lr<<"\tTrain loss: "<<L<<endl;
		double * pu2 = softmax_bp(u3, label, batch, ip2od);
		double * pd1 = inner_product_bp(u2, d1, w2, b2, pu2, lr, batch, ip1od, ip2od);
		double * pu1 = dropout_bp(pd1,p1,batch*ip1od);
		double * px1 = inner_product_bp(u1, data, w1, b1, pu1, lr, batch, x*y, ip1od);
		pos+=batch;
		pos%=n;
		lr*=gama;
		delete []px1;
		delete []pu1;
		delete []pd1;
		delete []pu2;
		delete []u3;
		delete []u2;
		delete []d1;
		delete []u1;
		delete []data;
		delete []label;
	}

	uint8_t * e = read_mnist_test_images(s, n, x, y);
	uint8_t * m = read_mnist_test_labels(s, n);
	int times=n/batch;
	pos=0;
	int num=0;
	while(times--){
		double * data = new double[batch*x*y];
		int * label = new int[batch];
		for(int i=0;i<batch*x*y;i++)
			data[i]=(int)e[i+pos*x*y]/255.0;
		for(int i=0;i<batch;i++)
			label[i]=(int)m[i+pos];
		double * u1 = inner_product(data,w1,b1,batch,x*y,ip1od);
		double * d1 = dropout(u1,p1,batch*ip1od);
		double * u2 = inner_product(d1,w2,b2,batch,ip1od,ip2od);
		double * u3 = softmax(u2,batch,ip2od);
		num += compute_correct_num(u3,label,batch,ip2od);
		pos+=batch;
		pos%=n;
		delete []u3;
		delete []u2;
		delete []d1;
		delete []u1;
		delete []data;
		delete []label;
	}
	cout<<"Test accuracy: "<<num/(n+0.0)<<endl;
	return 0;
}
