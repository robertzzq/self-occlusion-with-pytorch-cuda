#include "../utils/utils.h"

/**
 * @brief Whether a 2-dim vertex is in a 2-dim triangle
 */
template <typename scalar_t>
__device__ __forceinline__ int in_triangle_2d(
    scalar_t v1[2],
    scalar_t v2[2],
    scalar_t v3[2],
    scalar_t point[2]
)
{
    scalar_t sign_of_tri = (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v2[1] - v1[1]) * (v3[0] - v1[0]);
    scalar_t sign_of_ab = (v2[0] - v1[0]) * (point[1] - v1[1]) - (v2[1] - v1[1]) * (point[0] - v1[0]);
    scalar_t sign_of_ca = (v1[0] - v3[0]) * (point[1] - v3[1]) - (v1[1] - v3[1]) * (point[0] - v3[0]);
    scalar_t sign_of_bc = (v3[0] - v2[0]) * (point[1] - v3[1]) - (v3[1] - v2[1]) * (point[0] - v3[0]);

    int d1 = (sign_of_ab * sign_of_tri > 0);
    int d2 = (sign_of_ca * sign_of_tri > 0);
    int d3 = (sign_of_bc * sign_of_tri > 0);

    return d1 && d2 && d3;
}

/**
 * @brief Get the interpolation coeffs to interpolate v1, v2 and v3 to point.
 */
template <typename scalar_t>
__device__ __forceinline__ void get_interp_params(
    scalar_t v1[2],
    scalar_t v2[2],
    scalar_t v3[2],
    scalar_t point[2],
    scalar_t res[2]
)
{
    scalar_t u[2], v[2], w[2];
    scalar_t uu, uv, vv;
    scalar_t wu, wv;
    scalar_t r;

    sub2d(v2, v1, u);
    sub2d(v3, v1, v);
    sub2d(point, v1, w);

    uu = dot2d(u, u);
    uv = dot2d(u, v);
    vv = dot2d(v, v);

    wu = dot2d(w, u);
    wv = dot2d(w, v);

    r = 1.0 / (uv * uv - uu * vv);
    res[0] = (uv * wv - vv * wu) * r;
    res[1] = (uv * wu - uu * wv) * r;
    return;
}

/**
 * @brief Mapping 2-dim vertex on texture to 3-dim object
 * @param vertices: Input. All 3-dim triangles from the objects
 * @param tex_coords: Input. All texture triangles
 * @param point_2d: Input. Sampled 2-dim vertices to be mapped
 * @param res_3d: Output. Mapped 3-dim result
 * @param ind_1d: Output. Have the same length as res_3d. Indicates the triangle id for each mapped vertex
 * @param num_of_tri: number of triangles
 * @param num_of_samples: number of samples
 */
template <typename scalar_t>
__global__ void find_fragment_kernel(
    scalar_t* __restrict__ vertices,
    scalar_t* __restrict__ tex_coords,
    scalar_t* __restrict__ point_2d,
    scalar_t* __restrict__ res_3d,
    int* __restrict__ ind_1d,
    int num_of_tri,
    int num_of_samples
)
{
    int threadId_2D = threadIdx.x + threadIdx.y*blockDim.x;
    int blockId_2D = blockIdx.x + blockIdx.y*gridDim.x;
    int index = threadId_2D + (blockDim.x*blockDim.y)*blockId_2D;
    if (index < num_of_samples)
    {
        int i;
        scalar_t input_2d[2] = {
            point_2d[index * 2], point_2d[index * 2 + 1]
        };
        for (i = 0; i < num_of_tri; i++)
        {
            int i_v1 = i * 9;
            int i_v2 = i * 9 + 3;
            int i_v3 = i * 9 + 6;

            int i_t1 = i * 6;
            int i_t2 = i * 6 + 2;
            int i_t3 = i * 6 + 4;
            scalar_t t1[2] = {
                tex_coords[i_t1],
                tex_coords[i_t1 + 1]
            };
            scalar_t t2[2] = {
                tex_coords[i_t2],
                tex_coords[i_t2 + 1]
            };
            scalar_t t3[2] = {
                tex_coords[i_t3],
                tex_coords[i_t3 + 1]
            };

            if (in_triangle_2d(t1, t2, t3, input_2d))
            {
                scalar_t interp[2];
                get_interp_params(t1, t2, t3, input_2d, interp);
                scalar_t v1[3] = {
                    vertices[i_v1],
                    vertices[i_v1 + 1],
                    vertices[i_v1 + 2]
                };
                scalar_t v2[3] = {
                    vertices[i_v2],
                    vertices[i_v2 + 1],
                    vertices[i_v2 + 2]
                };
                scalar_t v3[3] = {
                    vertices[i_v3],
                    vertices[i_v3 + 1],
                    vertices[i_v3 + 2]
                };

                scalar_t u[3], v[3], u_interp[3], v_interp[3];
                sub(v2, v1, u);
                sub(v3, v1, v);
                mul(u, interp[0], u_interp);
                mul(v, interp[1], v_interp);

                scalar_t res_temp[3], temp[3];
                add(u_interp, v_interp, temp);
                add(v1, temp, res_temp);
                res_3d[index * 3] = res_temp[0];
                res_3d[index * 3 + 1] = res_temp[1];
                res_3d[index * 3 + 2] = res_temp[2];
                ind_1d[index] = i;
                return;
            }
            else
            {
                ind_1d[index] = -1;
            }
        }
    }
}

/**
 * @brief Texture mapping function. Calls cuda global kernel function find_fragment_kernel
 */
std::vector<torch::Tensor> texture_mapping(
    torch::Tensor vertices,
    torch::Tensor tex_coords,
    torch::Tensor sampling
)
{
    const auto num_of_triangles = vertices.size(0) / 3;

    const int threads = 1024;
    const dim3 blocks(32, 32);

    const auto num_of_samples = sampling.size(0);

    auto new_res_3d = torch::zeros({num_of_samples, 3}).to(torch::kCUDA);
    auto new_ind_1d = torch::zeros({num_of_samples}).to(torch::kInt32).to(torch::kCUDA);

    AT_DISPATCH_FLOATING_TYPES(vertices.type(), "find_fragment_kernel", ([&] {
    find_fragment_kernel<scalar_t><<<blocks, threads>>>(
        vertices.data<scalar_t>(),
        tex_coords.data<scalar_t>(),
        sampling.data<scalar_t>(),
        new_res_3d.data<scalar_t>(),
        new_ind_1d.data<int>(),
        num_of_triangles,
        num_of_samples);
    }));

    return {new_res_3d, new_ind_1d};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("texture_mapping", &texture_mapping, "texture_mapping (CUDA)");
}
