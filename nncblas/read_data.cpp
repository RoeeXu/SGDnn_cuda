/*****************************************************
* read_data.cpp
*
*
* cuda support c++ code
*
*
* Created by Roee Xu
*****************************************************/

#include <fstream>
#include <cstdint>

using namespace std;

int reverseint(int i){
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

int *read_mnist_test_labels(string pos, int &number_of_images){
    if(pos[pos.length()-1]!='/') pos+='/';
    pos+="t10k-labels.idx1-ubyte";
	ifstream file(pos, ios::binary);
    int magic_number = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseint(magic_number);
    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseint(number_of_images);
    int *label = new int[number_of_images];
    for(int i = 0;i < number_of_images;++i){
        unsigned char temp = 0;
        file.read((char*)&temp, sizeof(temp));
        label[i] = (int)temp;
    }
	file.close();
	return label;
}

int *read_mnist_train_labels(string pos, int &number_of_images){
    if(pos[pos.length()-1]!='/') pos+='/';
    pos+="train-labels.idx1-ubyte";
	ifstream file(pos, ios::binary);
	int magic_number = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseint(magic_number);
    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseint(number_of_images);
    int *label = new int[number_of_images];
    for(int i = 0;i < number_of_images;++i){
        unsigned char temp = 0;
        file.read((char*)&temp, sizeof(temp));
        label[i] = (int)temp;
    }
	file.close();
	return label;
}

int *read_mnist_test_images(string pos, int &number_of_images, int &n_rows, int &n_cols){
    if(pos[pos.length()-1]!='/') pos+='/';
    pos+="t10k-images.idx3-ubyte";
	ifstream file(pos, ios::binary);
	int magic_number = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseint(magic_number);
    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseint(number_of_images);
    file.read((char*)&n_rows, sizeof(n_rows));
    n_rows = reverseint(n_rows);
    file.read((char*)&n_cols, sizeof(n_cols));
    n_cols = reverseint(n_cols);
    int *data = new int[number_of_images * n_rows * n_cols];
    unsigned char temp;
    for(int i = 0;i < number_of_images;++i){
    	for(int j = 0;j < n_rows;++j){
    		for(int k = 0;k < n_cols;++k){
		        file.read((char*)&temp, sizeof(temp));
		        data[i * n_rows * n_cols + j * n_cols + k] = (int)temp;
    		}
    	}
    }
	file.close();
	return data;
}

int *read_mnist_train_images(string pos, int &number_of_images, int &n_rows, int &n_cols){
    if(pos[pos.length()-1]!='/') pos+='/';
    pos+="train-images.idx3-ubyte";
	ifstream file(pos, ios::binary);
	int magic_number = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseint(magic_number);
    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseint(number_of_images);
    file.read((char*)&n_rows, sizeof(n_rows));
    n_rows = reverseint(n_rows);
    file.read((char*)&n_cols, sizeof(n_cols));
    n_cols = reverseint(n_cols);
    int *data = new int[number_of_images * n_rows * n_cols];
    unsigned char temp;
    for(int i = 0;i < number_of_images;++i){
    	for(int j = 0;j < n_rows;++j){
    		for(int k = 0;k < n_cols;++k){
		        file.read((char*)&temp, sizeof(temp));
		        data[i * n_rows * n_cols + j * n_cols + k] = (int)temp;
    		}
    	}
    }
	file.close();
	return data;
}