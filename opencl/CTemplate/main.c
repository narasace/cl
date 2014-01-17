#include <stdio.h>
#include <stdlib.h>

#include <OpenCL/opencl.h>

extern const unsigned char kernels_src[];

#define FATAL(MSG) { fprintf(stderr, "Fatal error: " MSG "\n"); abort(); }

static void clpinfo(cl_platform_id platform)
{
        char name[256], vendor[256], version[256];
        
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(name), name, NULL);
        clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
        clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(version), version, NULL);
        
        fprintf(stdout, "Using %s platform, %s from %s\n", name, version, vendor);
}

static void cldinfo(cl_device_id device)
{
        cl_device_type type;
        cl_uint units, freq; cl_ulong mem, lmem;
        char name[256], vendor[256], version[256];
        
        clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
        clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
        clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(version), version, NULL);
        clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(units), &units, NULL);
        clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(freq), &freq, NULL);
        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, NULL);
        clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, NULL);
        
        fprintf(stdout, "Using %s%s (%u compute unit%s, %uMHz, %uMB/%ukB physical memory)\n",
                name, (type == CL_DEVICE_TYPE_GPU) ? " GPU" : "",
                units, (units > 1) ? "s" : "", freq,
                (cl_uint)(mem >> 20), (cl_uint)(lmem >> 10));
}

int main (int argc, const char *argv[])
{
        cl_uint i, p, platforms, devices;
        cl_platform_id platform[4];
        cl_device_id device[16];
        cl_command_queue q[16];
        cl_context context;
        cl_program program;
        
        const char *code[] = { kernels_src };
        
        cl_kernel kernel;
        cl_mem input, output;
        
        //const size_t wdim = 1, wsize[3] = {1024,1,1};
        const size_t wdim = 3, wsize[3] = {16,16,4};
        //const size_t wdim = 1, wsize[3] = {1<<27,1,1};
        const size_t chunk = wsize[0]*wsize[1]*wsize[2];
        
        cl_float data[chunk];
        //cl_float out[chunk];
        for (i=0; i < chunk; i++) data[i] = (i/chunk)*(1-i/chunk)*A;
        
        clGetPlatformIDs(4, platform, &platforms);
        if (!platforms) FATAL("No OpenCL runtime found")
        
        for (i = 0; i < 3; i++) {
                cl_device_type preference[3] = { CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ACCELERATOR, CL_DEVICE_TYPE_CPU };
                
                for (p = 0; p < platforms; p++) {
                        clGetDeviceIDs(platform[p], preference[i], 16, device, &devices); if (devices) break;
                }
                
                if (devices) break;
        }
        if (!devices) FATAL("No OpenCL devices found")
        
        clpinfo(platform[p]);
        
        context = clCreateContext(NULL, devices, device, NULL, NULL, NULL);
        if (!context) FATAL("Unable to create an OpenCL context")
        
        program = clCreateProgramWithSource(context, sizeof(code)/sizeof(const char *), code, NULL, NULL);
        if (!program) FATAL("Unable to create an OpenCL program")
        
        if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
                cl_build_status status; char log[1<<16];
                
                for (i = 0; i < devices; i++) {
                        clGetProgramBuildInfo(program, device[i], CL_PROGRAM_BUILD_STATUS, sizeof(status), &status, NULL);
                        clGetProgramBuildInfo(program, device[i], CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
                        
                        if (status != CL_BUILD_SUCCESS) { fprintf(stderr, "%s", log); abort(); }
                }
        }
        
        for (i = 0; i < devices; i++) {
                cldinfo(device[i]);
                
                q[i] = clCreateCommandQueue(context, device[i], 0, NULL);
                if (!q[i]) FATAL("Unable to create a dispatch queue")
        }
        
        
        input = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(data), data, NULL);
        output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(data), NULL, NULL);
        
        //for (i=0; i < 1024; i++) data[i] = 0.0;
        
        kernel = clCreateKernel(program, "b3copy", NULL);
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
        
        //for (i=0; i<1024; i++) 
        clEnqueueNDRangeKernel(q[0], kernel, wdim, NULL, wsize, NULL, 0, NULL, NULL);
        
        clFinish(q[0]);
        
        clEnqueueReadBuffer(q[0], output, CL_TRUE, 0, sizeof(data), data, 0, NULL, NULL);
        
        for (i=0; i < 1024; i+=64) printf("%f\n", data[i]);
        
        clReleaseKernel(kernel);
        clReleaseMemObject(output);
        clReleaseMemObject(input);
        for (i = 0; i < devices; i++)
                clReleaseCommandQueue(q[i]);
        clReleaseProgram(program);
        clReleaseContext(context);
        
        return 0;
}
