cu:
	nvcc nncu/nn_main.cu -lcudart -std=c++11 -o one

cpp:
	g++ -o one nncpp/nn_main_code.cpp -std=c++0x

cblas:
	g++ -c nncblas/read_data.cpp -std=c++0x
	g++ -c nncblas/main.cpp -lcblas
	g++ read_data.o main.o -o one -lcblas -std=c++0x
	rm -rf *.o

cublas:
	nvcc nncublas/main.cu -lcudart -lcublas -std=c++11 -o one

test:
	nvcc nncublas/test.cu -lcublas -std=c++11 -o test

clean:  
	rm -rf *.o one test
