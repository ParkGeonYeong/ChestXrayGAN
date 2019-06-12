"""
param = namedtuple('param',
                   'lr, batch, model, num_classes, epoch, '
                   'verbose_step, seed, augrate, display, mode, transfer, weight_decay_step, lr_decay_ratio, '
                   'template_shape, test_ratio')
"""

"""
if self.model == 'WGAN':
    self.d_lr = 0.00005
    self.g_lr = 0.00005
    self.beta1 = 0.9
    self.d_iter = 5
elif self.model == 'GAN':
    self.d_lr = 0.0002
    self.g_lr = 0.0002
    self.beta1 = 0.5
    self.d_iter = 1
"""
class Param(object):
    def __init__(self):
        self.model = 'WGAN'
        ##################
        self.d_lr = 0.0001
        self.g_lr = 0.0001
        self.beta1 = 0.5
        self.beta2 = 0.9
        self.d_iter = 5
        ##################
        self.batch_size = 32
        self.epoch = 30
        self.verbose_step = 10
        self.seed = 2019
        self.num_classes = 1
        self.augrate = 0.25
        self.display = True
        self.mode = 'train'
        self.gf_dim = 64 # First feature maps of generator convolution layer
        self.df_dim = 64  # First feature maps of discriminator convolution layer
        self.crop_width = 450
        self.resize = 128
        self.resize_ths = 900
        self.disease_group = ['No Finding', 'Cardiomegaly', 'Infiltration', 'Effusion']
        self.z_dim = 100
        self.save_step = 10
        self.sample_step = 50
        self.one_sided = 0.9
        self.standardize = False
        self.w_clip = 0.01
        self.d_iter = 5
        self.gp_lambda = 10.
        self.L1norm = True
hps = Param()
