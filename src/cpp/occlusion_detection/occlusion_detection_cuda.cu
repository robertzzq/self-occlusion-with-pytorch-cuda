#include "../utils/utils.h"

/**
 * @brief Given a triangle and a line segment, output the intersection and whether the line hits the triangle
 * @param v1: Input. 3-dim vector, one vertex of the triangle
 * @param v2: Input. 3-dim vector, one vertex of the triangle
 * @param v3: Input. 3-dim vector, one vertex of the triangle
 * @param origin: Input. 3d vector, start vertex of the line segment
 * @param dir: Input. 3d vector, normalized direction of the line segment
 * @param res: Output. 3d vector, containing information of the intersection
 * @return: whether the line hits the triangle
 */
template <typename scalar_t>
__device__ __forceinline__ scalar_t hit(
    scalar_t v1[3],
    scalar_t v2[3],
    scalar_t v3[3],
    scalar_t origin[3], scalar_t dir[3],
    scalar_t res[3]
)
{
    scalar_t e1[3], e2[3];
    sub(v2, v1, e1);
    sub(v3, v1, e2);

    scalar_t n[3];
    cross(e1, e2, n);

    scalar_t det = - dot(dir, n);
    scalar_t abs_det = det < -1e-6 ? -det : det;
    if (abs_det < 1e-6)
    {
        return 0.0f;
    }
    scalar_t inv_det = 1.0f / det;

    scalar_t a0[3], da0[3];
    sub(origin, v1, a0);
    cross(a0, dir, da0);

    scalar_t u = dot(e2, da0) * inv_det;
    scalar_t v = - dot(e1, da0) * inv_det;
    scalar_t t = dot(a0, n) * inv_det;
    res[0] = u;
    res[1] = v;
    res[2] = t;
    return (scalar_t)(abs_det >= 1e-6 && t >= 0.0 && u >= 0.0 && v >= 0.0 && (u+v) <= 1.0);
}

/**
 * @brief Perform view frustum clipping for each vertex
 * @param perspective_proj_mat: Input. Perspective projection matrix. 4x4
 * @param model_view_mat: Input. Model-View matrix. 4x4
 * @param samples: Input. Nx3 matrix. Sampled vertices
 * @param res_ind: Input/Output. Nx1. Infers which triangle a vertex belongs to. When -1, the vertex is out of view
 * @param num_of_samples: Input. N.
 */
template <typename scalar_t>
__global__ void frustum_clipping_kernel(
    scalar_t* __restrict__ perspective_proj_mat,
    scalar_t* __restrict__ model_view_mat,
    scalar_t* __restrict__ samples,
    int* __restrict__ res_ind,
    int num_of_samples
)
{
    int threadId_2D = threadIdx.x + threadIdx.y*blockDim.x;
    int blockId_2D = blockIdx.x + blockIdx.y*gridDim.x;
    int index = threadId_2D + (blockDim.x*blockDim.y)*blockId_2D;
    if (index < num_of_samples)
    {
        scalar_t current_sample[4] = {
            samples[index*3], samples[index*3+1], samples[index*3+2], 1.0
        };

        // model view projection
        scalar_t mv_proj_res[4];
        mv_proj_res[0] = model_view_mat[0]*current_sample[0] +
            model_view_mat[1]*current_sample[1] +
            model_view_mat[2]*current_sample[2] +
            model_view_mat[3]*current_sample[3];
        mv_proj_res[1] = model_view_mat[4]*current_sample[0] +
            model_view_mat[5]*current_sample[1] +
            model_view_mat[6]*current_sample[2] +
            model_view_mat[7]*current_sample[3];
        mv_proj_res[2] = model_view_mat[8]*current_sample[0] +
            model_view_mat[9]*current_sample[1] +
            model_view_mat[10]*current_sample[2] +
            model_view_mat[11]*current_sample[3];
        mv_proj_res[3] = model_view_mat[12]*current_sample[0] +
            model_view_mat[13]*current_sample[1] +
            model_view_mat[14]*current_sample[2] +
            model_view_mat[15]*current_sample[3];

        // perspective projection
        scalar_t pp_proj_res[4];
        pp_proj_res[0] = perspective_proj_mat[0]*mv_proj_res[0] +
            perspective_proj_mat[1]*mv_proj_res[1] +
            perspective_proj_mat[2]*mv_proj_res[2] +
            perspective_proj_mat[3]*mv_proj_res[3];
        pp_proj_res[1] = perspective_proj_mat[4]*mv_proj_res[0] +
            perspective_proj_mat[5]*mv_proj_res[1] +
            perspective_proj_mat[6]*mv_proj_res[2] +
            perspective_proj_mat[7]*mv_proj_res[3];
        pp_proj_res[2] = perspective_proj_mat[8]*mv_proj_res[0] +
            perspective_proj_mat[9]*mv_proj_res[1] +
            perspective_proj_mat[10]*mv_proj_res[2] +
            perspective_proj_mat[11]*mv_proj_res[3];
        pp_proj_res[3] = perspective_proj_mat[12]*mv_proj_res[0] +
            perspective_proj_mat[13]*mv_proj_res[1] +
            perspective_proj_mat[14]*mv_proj_res[2] +
            perspective_proj_mat[15]*mv_proj_res[3];

        // compare each element in pp_proj_res with pp_proj_res[3]
        scalar_t abs0 = pp_proj_res[0] < 0 ? -pp_proj_res[0] : pp_proj_res[0];
        scalar_t abs1 = pp_proj_res[1] < 0 ? -pp_proj_res[1] : pp_proj_res[1];
        scalar_t abs2 = pp_proj_res[2] < 0 ? -pp_proj_res[2] : pp_proj_res[2];
        scalar_t abs3 = pp_proj_res[3] < 0 ? -pp_proj_res[3] : pp_proj_res[3];
        if (!(abs0 <= abs3 && abs1 <= abs3 && abs2 <= abs3))
        {
            res_ind[index] = -1;
        }
    }
}

/**
 * @brief Perform occlusion detection for each vertex.
 * @param vertices: Input. All triangle fragments of the objects
 * @param origin: Input. 3-dim vector, start vertex of the line segment
 * @param dir: Input. Nx3, sampled vertices as destinations of line segments
 * @param dir_ind: Input. Nx1, Infers which triangle a vertex belongs to. When -1, the vertex is occluded
 * @param res: Output. Nx1. When 0, the vertex is occluded; when 1, the vertex is visible
 * @param num_of_tri: Input. number of triangles in the objects
 * @param num_of_samples: Input. N
 */
template <typename scalar_t>
__global__ void occlusion_detection_kernel(
    scalar_t* __restrict__ vertices,
    scalar_t* __restrict__ origin,
    scalar_t* __restrict__ dir,
    int* __restrict__ dir_ind,
    scalar_t* __restrict__ res,
    int num_of_tri,
    int num_of_samples
)
{
    int threadId_2D = threadIdx.x + threadIdx.y*blockDim.x;
    int blockId_2D = blockIdx.x + blockIdx.y*gridDim.x;
    int index = threadId_2D + (blockDim.x*blockDim.y)*blockId_2D;
    if (index < num_of_samples)
    {
        if (dir_ind[index] < 0)
        {
            res[index] = 0;
            return;
        }

        int i;
        scalar_t ori[3] = {
            origin[0], origin[1], origin[2]
        };
        scalar_t dst[3] = {
            dir[index*3], dir[index*3+1], dir[index*3+2]
        };

        scalar_t vec[3], vec_norm[3];
        sub(dst, ori, vec);
        norm(vec, vec_norm);
        scalar_t min_dist = dot(vec, vec);
        for (i = 0; i < num_of_tri; i++)
        {
            if (i == dir_ind[index])
            {
                continue;
            }
            int i_v1 = i * 9;
            int i_v2 = i * 9 + 3;
            int i_v3 = i * 9 + 6;
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
            scalar_t hit_res[3];
            scalar_t hit_ret = hit(v1, v2, v3, ori, vec_norm, hit_res);

            if (hit_ret > 0.5f)
            {
                scalar_t temp_dist = hit_res[2];
                if (temp_dist*temp_dist < min_dist)
                {
                    res[index] = 0;
                    return;
                }
            }
        }
        res[index] = 1;
    }
}

/**
 * @brief View frustum clipping function. Calls cuda global kernel function frustum_clipping_kernel
 */
torch::Tensor frustum_clipping(
    torch::Tensor prospective_proj_mat,
    torch::Tensor model_view_mat,
    torch::Tensor sampling,
    torch::Tensor tri_indices // int. for sampling
)
{
    const int threads = 1024;
    const dim3 blocks(32, 32);

    const auto num_of_samples = sampling.size(0);

    AT_DISPATCH_FLOATING_TYPES(prospective_proj_mat.type(), "frustum_clipping_kernel", ([&] {
    frustum_clipping_kernel<scalar_t><<<blocks, threads>>>(
        prospective_proj_mat.data<scalar_t>(),
        model_view_mat.data<scalar_t>(),
        sampling.data<scalar_t>(),
        tri_indices.data<int>(),
        num_of_samples);
    }));

    return tri_indices;
}

/**
 * @brief Occlusion function. Calls cuda global kernel function occlusion_detection_kernel
 */
torch::Tensor occlusion_detection(
    torch::Tensor vertices,
    torch::Tensor start,
    torch::Tensor sampling,
    torch::Tensor tri_indices // int. for sampling
)
{
    const auto num_of_triangles = vertices.size(0) / 3;

    const int threads = 1024;
    const dim3 blocks(32, 32);

    const auto num_of_samples = sampling.size(0);

    auto new_res = torch::zeros({num_of_samples, 1}).to(torch::kCUDA);

    AT_DISPATCH_FLOATING_TYPES(vertices.type(), "occlusion_detection_kernel", ([&] {
    occlusion_detection_kernel<scalar_t><<<blocks, threads>>>(
        vertices.data<scalar_t>(),
        start.data<scalar_t>(),
        sampling.data<scalar_t>(),
        tri_indices.data<int>(),
        new_res.data<scalar_t>(),
        num_of_triangles,
        num_of_samples);
    }));

    return new_res;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("occlusion_detection", &occlusion_detection, "occlusion_detection (CUDA)");
  m.def("frustum_clipping", &frustum_clipping, "frustum_clipping (CUDA)");
}
