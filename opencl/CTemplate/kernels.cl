/* $Id$ */

// test kernel doing nothing in particular
kernel void b3copy(global float *input, global float *output)
{
        size_t l = get_global_id(0), n = get_global_size(0);
	size_t k = get_local_id(0), kk = get_local_size(0);
         
        output[l] = input[l] + input[l];	// 7.8s
        
        /* uncomment printf for debugging */
        /*  printf("%i %i %i %i %f %f\n", l, n, k, kk, input[l], output[l]); */
}
