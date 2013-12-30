__kernel void kernels(__global float* a, __global float* b, __global float* c, __global float* d)
{

    unsigned int i = get_global_id(0);

    b[i] = 5*a[i];
    c[i] = b[i]*sin(8*M_PI*a[i]);
    d[i] = b[i]*cos(8*M_PI*a[i]);
}
