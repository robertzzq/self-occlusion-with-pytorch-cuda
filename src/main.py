import os
import sys
sys.path.append('../')
import time
import torch
from Entity.camera import Camera
from Entity.object import Object
from Entity.texture import Texture
from System.TextureMapping import sampling
from System.OcclusionDetection import occlusion, clipping
import numpy as np
from tqdm import tqdm


if __name__=="__main__":
    print("Initializing...")
    camera = Camera('../data/camera.json')
    object_3d = Object('../data/cube_plane.obj')
    texture = Texture()

    print("Processing...")
    time1 = time.time()
    # texture sampling to object sampling
    map_res = sampling(object_3d, texture)
    temp_v = map_res[0].cpu().numpy() # for .obj result output

    # view frustum clipping
    clip_res = clipping(map_res[0], map_res[1], camera)

    # occlusion
    occlude_res = occlusion(object_3d, camera, map_res[0], clip_res)
    time2 = time.time()

    # time calculation
    print('Processing time: %f ms' % ((time2 * 1000) - (time1 * 1000)))

    # save results
    print("Writing to 1024x1024 texture image.")
    texture.data = occlude_res.reshape((1024, 1024)) * 255
    texture.save('../temp/tex0.png')

    print("Writing to .obj file.")
    temp_data = occlude_res.cpu().numpy()
    with open('../temp/vertices0.obj', 'wt') as fp:
        for i in tqdm(range(len(temp_data))):
            if temp_data[i, 0] > 0.5:
                fp.write('v ' + str(temp_v[i, 0]) + ' ' + str(temp_v[i, 1]) + ' ' + str(temp_v[i, 2]) + '\n')

    print("done.")