import os
import sys
import json
import math
import torch
import torch.nn.functional as F


class Camera:
    def __init__(self, path : str):
        """
        Initialize camera from .json file.
        :param path: path
        """
        self.data = None
        try:
            with open(path, 'rt') as fp:
                self.data = json.load(fp)
        except Exception as exp:
            print('Error occurred initializing camera: ', str(exp))
            sys.exit(-1)
        print('Camera initialized: ', self.data)

        fov_x = self.data['Params']['h_fov'] * math.pi / 180
        fov_y = self.data['Params']['v_fov'] * math.pi / 180
        near = self.data['Params']['near']
        far = self.data['Params']['far']

        self.model_view_mat = self.get_model_view_matrix()

        self.perspective_proj = torch.FloatTensor([
            [1 / math.tan(fov_x / 2), 0, 0, 0],
            [0, 1 / math.tan(fov_y / 2), 0, 0],
            [0, 0, (far + near) / (far - near), -(2*near*far) / (far - near)],
            [0, 0, 1, 0]
        ]).cuda()

        self.position = torch.FloatTensor(self.data['Params']['eye']).cuda()

    def get_model_view_matrix(self):
        """
        Get the matrix which converts a coordinate from model(world in this program) to view
        :return: 4x4 pytorch tensor
        """
        lookat = torch.FloatTensor([self.data['Params']['lookat']])
        up = torch.FloatTensor([self.data['Params']['up']])
        eye = torch.FloatTensor([self.data['Params']['eye']])

        f = F.normalize(lookat - eye, p=2, dim=1)
        s = F.normalize(torch.cross(up, f))
        u = torch.cross(f, s)

        trans = -eye.transpose(0, 1)
        rotate = torch.cat((s, u, f), 0)
        trans = torch.mm(rotate, trans)

        temp = torch.cat((rotate, trans), 1)

        return torch.cat((temp, torch.FloatTensor([[0, 0, 0, 1]])), 0).cuda()


if __name__=="__main__":
    cam = Camera('../../data/camera.json')
    print(cam.model_view_mat)
    print(cam.perspective_proj)
