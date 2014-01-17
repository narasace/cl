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

#define DATA_SIZE (65536)

const char *KernelSource = "\n" \
"__kernel void wave(                                                    \n" \
"   __global float* inputx,                                             \n" \
"   __global float* inputt,                                             \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   int j = 0;                                                          \n" \
"   while(j < count){                                                   \n" \
"   if(i < count){                                                      \n" \
"       output[i] = output[i] + inputx[i] * inputt[i];}                 \n" \
"   j = j+1;}                                                           \n" \
"}                                                                      \n" \
"\n";
 

int main(int argc, char** argv)
{
    
    clock_t start = clock();
    int err;                            // error code returned from api calls
    float fofx[DATA_SIZE];
    float foft[DATA_SIZE]; 
    float results[DATA_SIZE];           // results returned from device
    unsigned int correct;               // number of correct results returned

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    cl_platform_id platform_id;
    cl_uint num_id;
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    cl_mem inputx;                      // device memory used for the inputa array 
    cl_mem inputt;                      // device memory used for the inputb array
    cl_mem output;                      // device memory used for the output array

    float alpha = 0.999;                // make sure nothing will go unbounded
    float dx = 1.0 / DATA_SIZE;         // size of spatial spacing
    float dt = dx / alpha;              // size of time steps

    // int nx = pow(2, 8);                 // spatial grid for output
    // int nt = pow(2, 3);                 // time grid for output

    int i = 0;
    // unsigned int timestep = DATA_SIZE;
    unsigned int count = DATA_SIZE;
    for(i = 0; i < count; i++){

        fofx[i] = i*dx;
        foft[i] = i*dt;
    }
    //    printf("%f, %f\n", fofx[i], foft[i]);}

    

    err = clGetPlatformIDs(1, &platform_id, &num_id);
    if (err != CL_SUCCESS)
    {
        printf("Failed to get platform ID!\n");
        return EXIT_FAILURE;
    }
    // printf("Platform ID: %i\n", platform_id);
    printf("ID of the platform: %i\n", num_id);

    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
    // printf("Device ID: " + device_id + "\n");
    printf("Demo started here!\n");


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
        char buffer[65536];
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


    inputx = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    inputt = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    output = clCreateBuffer(context,  CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);

    if (!inputx || !inputt || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }
    printf("clCreateBuffer\n");    


    err = clEnqueueWriteBuffer(commands, inputx, CL_TRUE, 0, sizeof(float) * count, fofx, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(commands, inputt, CL_TRUE, 0, sizeof(float) * count, foft, 0, NULL, NULL);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
    printf("clEnqueueWriteBuffer\n");


    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputx);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputt);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &count);

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

    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);

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

    // correct = 0;
    // float testout[count];

    // for(i = 0; i < count; i++)
    // {
    //     testout[i] = pow(data[i],data[i]);
    //     if(results[i] == testout[i]){
    //         correct++;}
    //     printf("%f, %f\n", testout[i], results[i]);
    // }

    printf("Computed values:\n");
    
    int k = 0;
    while(k < count)
    {
        printf("%f\n",results[k]);
        k = k + 1024;
    }


    clReleaseMemObject(inputx);
    clReleaseMemObject(inputt);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    clock_t end = clock();

    printf("Time took: %f seconds\n", ((double)end - (double)start)*1.0e-6);

    return 0;
}
