/*****************************************************
* random.h
*
*
* random function
*
*
* Created by Roee Xu
*****************************************************/

#include <cmath>
#include <cstdlib>

double gaussrand(double E, double V)
{
    static double V1, V2, S;
    static int phase = 0;
    double X;
    if ( phase == 0 ) {
        do {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);
        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);
    phase = 1 - phase;
    X = X * V + E;
    return X;
}

int boolrand(double p){
	double x = (double)rand() / RAND_MAX;
	x+=p;
	return floor(x);
}