import argparse
import os
from util import util
import torch

class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self):
        # 3D GAN

        import easydict
        self.opt = easydict.EasyDict({
            # ---------------------------------------
            # Generator - 3DGAN
            "obj":'table',
            "workers": 2,
            "cuda": 1,
            "batch_size": 64,
            "imageSize":64,
            "localSize":32,
            "hole_min":12,
            "hole_max":18,
            "ndf":64,
            "niter":100,
            "preniter":20,
            "lr":0.0001,
            "alpha":0.01,
            "beta1":0.5,
            "beta2": 0.99,
            "manualSeed":None,
            "LeakyReLu": True,
            "leak_value": 0.2,
            "bias": False,
            "G_output_activation":nn.Tanh(),
            "soft_label": True
            # --------------------------------------

            # Discriminator - MeshCNN
            # Trainoptions
            # "batch_size":16,
            "print_freq": 10,  # frequency of showing training results on console
            "save_latest_freq": 250,  # frequency of saving the latest results
            "run_test_freq": 1,  # requency of runninglr_policy test in training script
            "continue_train": False,  # continue training: load the latest model
            "epoch_count": 1,
            # the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...
            "phase": 'train',  # 'train, val, test, etc'
            "which_epoch": 'latest',  # which epoch to load?
            "niter_decay": 2000,  # # of iter to linearly decay learning rate to zero
            # "beta1": 0.9,  # momentum term of adam
            "lr_policy": 'lambda',  # learning rate policy: lambda|step|plateau
            "lr_decay_iters": 50,  # multiply by a gamma every lr_decay_iters iterations
            "input_nc": 5,
            # data augmentation stuff
            "num_aug": 10,  # # of augmentation files
            "scale_verts": True,  #
            "slide_verts": 0,  #
            "flip_edges": 0,  #
            "no_vis": True,  # will not use tensorboard
            "verbose_plot": True,  # plots network weights, etc.
            "is_train": True,
            "ninput_edges": 750,
            "resblocks": 0,
            "fc_n": 100,
            "ncf": [64, 128, 256, 256],
            "pool_res":[600, 450, 300, 180],
            "norm": 'batch',  # batch normalization adopted
            "num_groups": 16,
            "init_type": 'normal',
            "init_gain": 0.02,
            "num_threads": 3,
            "gpu_ids": str('0'),
            "name": 'debug',
            "checkpoints_dir": MODEL_DIR + obj,
            "serial_batches": True,
            "seed": None,
            "export_folder": EXPORT_DIR,
            "initialized": True,
            # For mesh faces extraction
            "threshold": 0.5
            # Contour value to search for isosurfaces in volume. If not given or None, the average of the min and max of vol is used.
            "D_label": None
        })

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.opt
        # self.opt, unknown = self.parser.parse_known_args()
        # self.opt.is_train = self.is_train   # train or test
        self.is_train = self.opt.is_train

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        if self.opt.seed is not None:
            import numpy as np
            import random
            torch.manual_seed(self.opt.seed)
            np.random.seed(self.opt.seed)
            random.seed(self.opt.seed)

        if self.opt.export_folder:
            self.opt.export_folder = self.opt.export_folder  # os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.export_folder)
            mkdir(self.opt.export_folder)

        if self.is_train:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

            # save to the disk
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            mkdir(expr_dir)

            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt

