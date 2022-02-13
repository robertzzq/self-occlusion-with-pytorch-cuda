import os
import sys

sys.path.append('../../')
sys.path.append('../Entity/')
import torch
from src.Entity.object import Object
from src.Entity.texture import Texture
import texture_mapping


def sampling(object_3d: Object, texture: Texture):
    """
    Map sampled 2d vertices on texture to 3d object
    :param object_3d: All 3-dim triangles from the objects
    :param texture:  All texture triangles
    :return: List<Tensor>. The first is a mapped 3-dim result; The second has the same length as the 1st, indicating the triangle id for each mapped vertex
    """
    return texture_mapping.texture_mapping(object_3d.vertices, object_3d.tex_coords, texture.samples)
