#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DATA_SIZE (65536)
int main()
{
	clock_t start = clock();

	float fofx[DATA_SIZE];
	float foft[DATA_SIZE];
	float results[DATA_SIZE];

	float alpha = 0.999;
	float dx = 1.0 / DATA_SIZE;
	float dt = dx;

	int count = DATA_SIZE;

	int i = 0;
	for(i = 0; i < count; i++){
		fofx[i] = i * dx;
		foft[i] = i * dt;
	//	results[i] = 0;
	}

	int j = 0;
	while(j < count){
		for(i = 0; i < count; i++){
			results[i] = results[i] + fofx[i] * foft[i];
		}
		j = j + 1;
	}

	int k = 0;
	while(k < count){
		printf("%f\n",results[k]);
		k = k + 1024;
	}

	clock_t end = clock();

	printf("Time took: %f seconds\n", ((double)end - (double)start)*1.0e-6);
}