#ifndef _SELF_OCCLUSION_UTILS_H_
#define _SELF_OCCLUSION_UTILS_H_

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <math.h>

//////////////////////////// 2d
template <typename scalar_t>
__device__ __forceinline__ scalar_t dot2d(
    scalar_t x[2],
    scalar_t y[2]
)
{
    return x[0]*y[0] + x[1]*y[1];
}

template <typename scalar_t>
__device__ __forceinline__ void add2d(
    scalar_t x[2],
    scalar_t y[2],
    scalar_t res[2]
)
{
    res[0] = x[0] + y[0];
	res[1] = x[1] + y[1];
	return;
}

template <typename scalar_t>
__device__ __forceinline__ void sub2d(
    scalar_t x[2],
    scalar_t y[2],
    scalar_t res[2]
)
{
    res[0] = x[0] - y[0];
	res[1] = x[1] - y[1];
	return;
}

template <typename scalar_t>
__device__ __forceinline__ void mul2d(
    scalar_t x[2],
    scalar_t y,
    scalar_t res[2]
)
{
    res[0] = x[0] * y;
	res[1] = x[1] * y;
	return;
}

template <typename scalar_t>
__device__ __forceinline__ void div2d(
    scalar_t x[2],
    scalar_t y,
    scalar_t res[2]
)
{
    res[0] = x[0] / y;
	res[1] = x[1] / y;
	return;
}

template <typename scalar_t>
__device__ __forceinline__ void norm2d(
    scalar_t x[2],
    scalar_t res[2]
)
{
    scalar_t sqrt_len = sqrt(dot2d(x, x));
    div2d(x, sqrt_len, res);
	return;
}

//////////////////////////// 3d

template <typename scalar_t>
__device__ __forceinline__ void cross(
    scalar_t x[3],
    scalar_t y[3],
    scalar_t res[3]
)
{
    res[0] = x[1]*y[2] - x[2]*y[1];
    res[1] = x[2]*y[0] - x[0]*y[2];
    res[2] = x[0]*y[1] - x[1]*y[0];
    return;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t dot(
    scalar_t x[3],
    scalar_t y[3]
)
{
    return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
}

template <typename scalar_t>
__device__ __forceinline__ void add(
    scalar_t x[3],
    scalar_t y[3],
    scalar_t res[3]
)
{
    res[0] = x[0] + y[0];
	res[1] = x[1] + y[1];
	res[2] = x[2] + y[2];
	return;
}

template <typename scalar_t>
__device__ __forceinline__ void sub(
    scalar_t x[3],
    scalar_t y[3],
    scalar_t res[3]
)
{
    res[0] = x[0] - y[0];
	res[1] = x[1] - y[1];
	res[2] = x[2] - y[2];
	return;
}

template <typename scalar_t>
__device__ __forceinline__ void mul(
    scalar_t x[3],
    scalar_t y,
    scalar_t res[3]
)
{
    res[0] = x[0] * y;
	res[1] = x[1] * y;
	res[2] = x[2] * y;
	return;
}

template <typename scalar_t>
__device__ __forceinline__ void div(
    scalar_t x[3],
    scalar_t y,
    scalar_t res[3]
)
{
    res[0] = x[0] / y;
	res[1] = x[1] / y;
	res[2] = x[2] / y;
	return;
}

template <typename scalar_t>
__device__ __forceinline__ void norm(
    scalar_t x[3],
    scalar_t res[3]
)
{
    scalar_t sqrt_len = sqrt(dot(x, x));
    div(x, sqrt_len, res);
	return;
}

#endif
