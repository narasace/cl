#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Solves 1-D wave equation Utt = v^2(Uxx) with initial wave function U = x*(L-x)*A

int main()
{

    clock_t start = clock();         // Start the timer
    float L = 1;                     // Spatial range
    int sStep = 131072;              // Number of spatial intervals, spatial steps
    int tStep = 131072;              // Number of time steps for the evolution
    float alpha = 0.999;             // To preserve stability
    float dx = L / sStep;            // Calculate spatial interval
    float dt = dx / alpha;           // Calculate time steps
    float v = 1;                     // Velocity of the wave
    float A = 3;                     // Amplitude
    float c = v * (dt / dx);         // Courant number

    float U[sStep];                  // Initialize U

    float sten[sStep];              // Intermediate stencil in memory
    float results[sStep];            // Results in memory

    int i = 0;

    // Set the initial values
    for(i = 0; i <= sStep; i++)
    {
	    U[i] = (i*dx)*(L-i*dx)*A;   // This enforces the boundary to be zero
	    // printf("%f\n", U[i]);
    }

    
    // Now claculate the time evolution

    // The first step

    results[0] = 0;
    results[sStep] = 0;

    int j = 1;
    while(j < sStep)
    {
    	results[j] = U[j] - 0.5*pow(alpha,2)*(U[j+1] - 2*U[j] + U[j-1]);
    	j = j+1;  
    }

    // The following steps

    int k = 2;
    while(k <= tStep)
    {
    	sten[0] = 0;
    	sten[sStep] = 0;
    	for(i = 1; i < sStep; i++)
    	{
    		sten[i] = U[i];
    	}

    	U[0] = 0;
    	U[sStep] = 0;
    	for(i = 1; i < sStep; i++)
    	{
    		U[i] = results[i];
    	}

    	results[0] = 0;
    	results[sStep] = 0;
    	for(i = 1; i < sStep; i++)
    	{
    		results[i] = -sten[i] + 2*U[i] + pow(alpha,2)*(U[i+1] - 2*U[i] + U[i-1]);
    	}
    	k = k + 1;
    	// printf("%f\n", U[sStep/2]);
    }

    clock_t end = clock();

    printf("It took %f seconds. \n", ((double)end - (double)start)*1.0e-6);

}
