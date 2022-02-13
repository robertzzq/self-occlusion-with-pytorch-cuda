import os
import sys

sys.path.append('../../')
sys.path.append('../Entity/')
import torch
from src.Entity.camera import Camera
from src.Entity.object import Object
import occlusion_detection


def clipping(samples, indices_tri, camera: Camera):
    """
    view frustum clipping
    :param samples: Input. Nx3 matrix. Sampled vertices
    :param indices_tri: In/Out. Nx1. Infers which triangle a vertex belongs to. When -1, the vertex is out of view
    :param camera: Input. camera
    :return: returns a copy of indices_tri
    """
    new_indices = occlusion_detection.frustum_clipping(camera.perspective_proj, camera.model_view_mat, samples, indices_tri)
    return new_indices


def occlusion(object_3d: Object, camera: Camera, samples_3d, samples_3d_ind_tri):
    """
    self occlusion detection
    :param object_3d: Input. 3d object
    :param camera: Input. camera
    :param samples_3d: Input. Nx3 matrix. Sampled vertices
    :param samples_3d_ind_tri: Input. Nx1. Infers which triangle a vertex belongs to. When -1, the vertex is invisible
    :return: Indicator for each sampled vertex. 0 indicates invisible, 1 indicates visible
    """
    return occlusion_detection.occlusion_detection(object_3d.vertices, camera.position, samples_3d, samples_3d_ind_tri)


if __name__ == "__main__":
    cam = Camera('../../data/camera.json')
    dummy = torch.FloatTensor([[0, 0, 1000]]).cuda()
    indices = torch.IntTensor([[1]]).cuda()
    res = clipping(dummy, indices, cam)
    print(res)
