import os
import sys
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


class Texture:
    def __init__(self):
        """
        Initialize a 1024x1024 texture image from file
        """
        self.data = torch.zeros((1024, 1024), dtype=torch.float).cuda()
        print('Texture image initialized, size: ', self.data.shape)

        temp_samples = []
        for i in range(1024):
            for j in range(1024):
                temp_samples.append([j, 1024 - i])
        self.samples = torch.FloatTensor(temp_samples).cuda() / 1024

    def save(self, path: str):
        """
        Save the texture as an image.
        :param path: output path
        :return: void
        """
        assert (not (self.data is None))
        temp_tensor = self.data.cpu().clone()
        temp_tensor = temp_tensor.type(torch.uint8).numpy()
        image = Image.fromarray(temp_tensor)
        image.save(path)
        return


if __name__ == "__main__":
    tex = Texture()
    print(tex.samples)
    # tex.save('../../temp/tex.png')
