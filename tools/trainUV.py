import argparse
from omegaconf import OmegaConf

import os, sys
sys.path.append(os.getcwd())

from lib.trainerUV import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    gui = Trainer(opt)

    if opt.gui:
        gui.render()
    else:
        gui.train(opt.iters)
