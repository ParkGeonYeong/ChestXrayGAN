"""
param = namedtuple('param',
                   'lr, batch, model, num_classes, epoch, '
                   'verbose_step, seed, augrate, display, mode, transfer, weight_decay_step, lr_decay_ratio, '
                   'template_shape, test_ratio')
"""


class Param(object):
    def __init__(self):
        self.lr = 0.0001
        self.beta1 = 0.5
        self.batch_size = 128
        self.epoch = 40
        self.verbose_step = 10
        self.seed = 2019
        self.num_classes = 1
        self.augrate = 0.25
        self.display = True
        self.mode = 'train'
        self.feature_map = 32
        self.crop_width = 400
        self.resize = 256
        self.resize_ths = 900
        self.disease_group = ['No Finding', 'Infiltration', 'Effusion', 'Atelectasis', 'Nodule']
        self.z_dim = 100
        self.save_step = 10
        self.sample_step = 30
hps = Param()