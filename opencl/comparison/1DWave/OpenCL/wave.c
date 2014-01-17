#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>

// Solve for 1-d wave equation Utt = v^2(Uxx)
// Same iteration as in C but in a parallel framework

#define DATA_SIZE (256)

const char *KernelSource = "\n" \
"__kernel void wave(                                                                                      \n" \
"   __global float* inputa,                                                                               \n" \
"   __global float* inputb,                                                                               \n" \
"   __global float* output,                                                                               \n" \
"   const unsigned int count)                                                                             \n" \
"{                                                                                                        \n" \
"   int i = get_global_id(0);                                                                             \n" \
"   int j = 0;                                                                                            \n" \
"   while (j < count){                                                                                    \n" \
"                                                                                                         \n" \
"       output[0] = 0;                                                                                  \n" \
"       output[i] = -inputb[i] + 2*inputa[i] + 0.9999*0.9999*(inputa[i+1] - 2*inputa[i] + inputa[i-1]);   \n" \
"       printf(\"%zu %zu %f\\n\", j, i, 0.95);                                                         \n" \
"       //barrier(CLK_GLOBAL_MEM_FENCE);                                                                    \n" \
"       //inputb[i] = inputa[i];                                                                            \n" \
"       //barrier(CLK_GLOBAL_MEM_FENCE);                                                                    \n" \
"       //inputa[i] = output[i];                                                                           \n" \
"                                                                           \n" \
"   }                                                                                             \n" \
"}                                                                                                        \n" \
"\n";
 
int main(int argc, char** argv)
{
    
    clock_t start = clock();
    int err;                            // error code returned from api calls
    int L = 1;                          // range
    float U[DATA_SIZE];                 // initialize U
    float sten[DATA_SIZE];              // stencil for memory
    float results[DATA_SIZE];           // results returned from device
    float A = 3;
    unsigned int correct;               // number of correct results returned

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    cl_platform_id platform_id;         // platform ID
    cl_uint num_id;                     // numerical value
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    cl_mem inputa;                      // device memory used for the U array 
    cl_mem inputb;                      // device memory used for the stencil array
    cl_mem output;                      // device memory used for the output array

    float alpha = 0.9999;               // make sure nothing will go unbounded
    float dx = 1.0 / DATA_SIZE;         // size of spatial spacing
    float dt = dx / alpha;              // size of time steps

    int i = 0;
    
    size_t count = DATA_SIZE;


    // Fill in U array at time = 0

    for(i = 0; i <= count; i++){
        U[i] = (i*dx)*(L-i*dx)*A;
    }

    // Calculate stencil at x = 0

    sten[0] = 0;
    sten[count] = 0;

    for(i = 1; i < count; i++){
    	sten[i] = U[i] - 0.5*pow(alpha,2)*(U[i+1] - 2*U[i] + U[i-1]);
    }



    err = clGetPlatformIDs(1, &platform_id, &num_id);
    if (err != CL_SUCCESS)
    {
        printf("Failed to get platform ID!\n");
        return EXIT_FAILURE;
    }
    printf("ID of the platform: %i\n", num_id);

    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
    printf("Program starte from here!\n");


    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }
    printf("clCreateContext\n");
 

    commands = clCreateCommandQueue(context, device_id, 0, &err);

    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }
    printf("clCreateCommandQueue\n");

 
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);

    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }
    printf("clCreateProgramWithSource\n");

 
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[262144];
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
    printf("clBuildProgram\n");

 
    kernel = clCreateKernel(program, "wave", &err);

    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }
    printf("clCreateKernel\n");


    inputa = clCreateBuffer(context,  CL_MEM_READ_WRITE,  count*sizeof(cl_float), NULL, NULL);
    inputb = clCreateBuffer(context,  CL_MEM_READ_WRITE,  count*sizeof(cl_float), NULL, NULL);
    output = clCreateBuffer(context,  CL_MEM_READ_WRITE,  count*sizeof(cl_float), NULL, NULL);

    if (!inputa || !inputb || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }
    printf("clCreateBuffer\n");    


    err = clEnqueueWriteBuffer(commands, inputa, CL_TRUE, 0, count*sizeof(cl_float), U, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(commands, inputb, CL_TRUE, 0, count*sizeof(cl_float), sten, 0, NULL, NULL);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
    printf("clEnqueueWriteBuffer\n");


    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(inputa), &inputa);
    err |= clSetKernelArg(kernel, 1, sizeof(inputb), &inputb);
    err |= clSetKernelArg(kernel, 2, sizeof(output), &output);
    //err |= clSetKernelArg(kernel, 3, sizeof(count), &count);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

 
    global = count;

    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);

    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }



    clFinish(commands);


    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL );  

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    int j = 0;

    while(j < count){
    	printf("%f\n", results[j]);
    	j = j + 512;
    }


    clReleaseMemObject(inputa);
    clReleaseMemObject(inputb);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    clock_t end = clock();

    printf("Time took: %f seconds\n", ((double)end - (double)start)*1.0e-6);

    return 0;
}

// kernel ND range, try trivial kernel, then check buffer, then check the order by printf() the index
// Argument passing
// kernel argument setup
// Initialization
// try with empty kernel, print out the global IDs, then try with one argument, then with some more
// 
// pay close attentino to the size of types of numerical variables

