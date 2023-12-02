import argparse
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import cv2

import os, sys
sys.path.append(os.getcwd())

from lib.trainer import Trainer
from lib.dataset.ZJU import ZJU


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    ZJU = ZJU(opt)
    data = ZJU[11800]
    
    vertices = data['vertices']
    K = data['K']
    vertices_2d = (K @ vertices.T).T
    vertices_2d = vertices_2d[:, :2] / vertices_2d[:, 2:]
    img = data['image']
    for vertex in vertices_2d:
        img[int(vertex[1]), int(vertex[0])] = [255, 255, 255]
    cv2.imwrite('test.jpg', img)
