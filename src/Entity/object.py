import os
import sys
import logging
import torch
import numpy as np


class Object:
    def __init__(self, path: str):
        """
        Initialize from .obj file (only supports triangular mesh)
        :param path: string, .obj file path
        """
        if not (os.path.exists(path) and os.path.isfile(path)):
            logging.log(logging.ERROR, '\"path\" does not exist or is not a file.')
            sys.exit(-1)
        temp_vertices = []
        temp_tex_coords = []
        temp_v_indices = []
        temp_t_indices = []
        with open(path, 'rt') as fp:
            lines = fp.readlines()
            for line in lines:
                line_array = line.strip().split(' ')
                if line_array[0] == 'v':                    # vertex
                    temp_vertices.append([float(line_array[1]), float(line_array[2]), float(line_array[3])])
                elif line_array[0] == 'vt':                 # texture coordinate
                    temp_tex_coords.append([float(line_array[1]), float(line_array[2])])
                elif line_array[0] == 'f':                  # face
                    v1 = line_array[1].split('/')
                    v2 = line_array[2].split('/')
                    v3 = line_array[3].split('/')
                    assert(len(v1) >= 2 and len(v2) >= 2 and len(v3) >= 2)
                    temp_v_indices.append([int(v1[0]) - 1, int(v2[0]) - 1, int(v3[0]) - 1])
                    temp_t_indices.append([int(v1[1]) - 1, int(v2[1]) - 1, int(v3[1]) - 1])
        self.vertices = []
        self.tex_coords = []
        for i in range(len(temp_v_indices)):
            self.vertices.append(temp_vertices[temp_v_indices[i][0]])
            self.vertices.append(temp_vertices[temp_v_indices[i][1]])
            self.vertices.append(temp_vertices[temp_v_indices[i][2]])
            self.tex_coords.append(temp_tex_coords[temp_t_indices[i][0]])
            self.tex_coords.append(temp_tex_coords[temp_t_indices[i][1]])
            self.tex_coords.append(temp_tex_coords[temp_t_indices[i][2]])

        self.vertices = torch.FloatTensor(self.vertices).cuda()
        self.tex_coords = torch.FloatTensor(self.tex_coords).cuda()


if __name__=="__main__":
    obj = Object('../../data/cube.obj')
    print(obj.vertices)
    print(obj.tex_coords)
