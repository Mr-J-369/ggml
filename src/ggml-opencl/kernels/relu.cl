#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// relu
//------------------------------------------------------------------------------
kernel void kernel_relu(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    dst[get_global_id(0)] = fmax(0.0f, src0[get_global_id(0)]);
}

kernel void kernel_leaky_relu(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        float slope
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    float x = src0[get_global_id(0)];
    dst[get_global_id(0)] = fmax(x, 0.0f) + fmin(x, 0.0f) * slope;
}

