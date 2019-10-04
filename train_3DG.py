from __future__ import print_function
import easydict
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import sys
sys.path.insert(0, '/home/yina_lin_18/3DMeshConvGan/3DGAN/GAN_util/')
from GAN_util.util import *
from GAN_util.logger import Logger
from GAN_util.models import *

from torch.autograd import Variable

# IF RUN ON COLAB or ipython: remove the hash tags below
# -------------------------------------------------------
#! npm install -g localtunnel
#! npm i -g npm
#get_ipython().system_raw('python3 -m pip install visdom')
#get_ipython().system_raw('python3 -m visdom.server -port 6006 >> visdomlog.txt 2>&1 &')
#get_ipython().system_raw('lt --port 6006 >> url.txt 2>&1 &')
#get_ipython().system_raw('ssh -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -R trial:80:localhost:6006 serveo.net &')
#import time
#time.sleep(5)
#! cat url.txt
#import visdom
#time.sleep(5)
#vis = visdom.Visdom(port='6006')
#print(vis)
#time.sleep(3)
#vis.text('testing')
#! cat visdomlog.txt

#print('Visdom view: {}'.format('https://trial.serveo.net/'))

# ---------------------------------------------------------


# ---------------------------------------------------
# OPTIONS:
# ---------------------------------------------------
opt = easydict.EasyDict({
    "obj":'piano',
    "workers": 2,
    "cuda": 1,
    "batch_size": 64,
    "imageSize":64,
    "localSize":32,
    "hole_min":12,
    "hole_max":18,
    "ndf":64,
    "niter":100,
    "preniter":0,
    "lr":0.0001,
    "alpha":0.01,
    "beta1":0.5,
    "beta2": 0.99,
    "manualSeed":None,
    "LeakyReLu": True,
    "leak_value": 0.2,
    "bias": False,
    "G_output_activation":nn.Tanh(),
    "soft_label": False
})


print(opt)

# ---------------------------------------------------
# SETUPS:
# ---------------------------------------------------

# DATA DIR:
DATA_DIR = '/home/yina_lin_18/3DMeshConvGan/data/'
MODEL_DIR = '/home/yina_lin_18/3DMeshConvGan/3DGAN/models/'
LOG_DIR = '/home/yina_lin_18/3DMeshConvGan/3DGAN/logs/' + opt.obj
GEN_DATA_DIR = '/home/yina_lin_18/3DMeshConvGan/3DGAN/gen_data/' + opt.obj

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

device = torch.device("cuda:0" if opt.cuda else "cpu")
ndf = int(opt.ndf)
nc = 3
obj = opt.obj
obj_ratio = 0.7

# logger = Logger(LOG_DIR)

if not os.path.exists('models'):
  os.makedirs('models')




# Computes and stores the average and current value
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, curr_epoch):
    torch.save(state, MODEL_DIR + '/netG_e%d.pth.tar' % (curr_epoch))

# ---------------------------------------------------
# Initialize Generator and Discriminator:
# ---------------------------------------------------

netG = Generator(opt).to(device)
weights_init(netG)
print(netG)

netD = Discriminator(opt).to(device)
weights_init(netD)
print(netD)

# ---------------------------------------------------
# Load all the data:
# ---------------------------------------------------

volumes = getAll(obj=obj, train=True, obj_ratio=obj_ratio)
print('Using ' + obj + ' Data')
volumes = volumes[..., np.newaxis].astype(np.float)
data = torch.from_numpy(volumes)
data = data.permute(0, 4, 1, 2, 3)
data = data.type(torch.FloatTensor)

# ---------------------------------------------------
# Define Loss criterion:
# ---------------------------------------------------

criterion = nn.BCELoss()
criterion_G = nn.MSELoss()

# ---------------------------------------------------
# Optimizers:
# ---------------------------------------------------

optG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
optD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))


# ---------------------------------------------------
# Setup Loss storages:
# ---------------------------------------------------

lossD_all = AverageMeter()
lossG_all = AverageMeter()


loss_dict = {
    'D loss':[],
    'G loss':[]
}

# ---------------------------------------------------
# Setup Labels:
# ---------------------------------------------------

if opt.soft_label:
    real_label = Variable(torch.Tensor(opt.batch_size).uniform_(0.7, 1.2).cuda())
    fake_label = Variable(torch.Tensor(opt.batch_size).uniform_(0, 0.3).cuda())
else:
    real_label = torch.full((opt.batch_size,), 1, device=device)
    fake_label = torch.full((opt.batch_size,), 0, device=device)

# ---------------------------------------------------
# Start Training:
# ---------------------------------------------------
n_batches = int(data.size(0) / opt.batch_size)

for epoch in range(opt.niter):
    time0 = time.time()

    for i in range(n_batches):

        batch_data = data[i * opt.batch_size:(i * opt.batch_size + opt.batch_size)]

        real_data = batch_data.to(device)

        # Add noise:
        masks = cuboid_noise(opt)
        masks = torch.from_numpy(masks)
        masks = masks.type(torch.FloatTensor).cuda()
        masked_data = real_data + masks
        masked_data[masked_data > 1] = 1

        # WARM UP GENERATOR:
        if epoch <= opt.preniter:
            optG.zero_grad()
            fake_data = netG(masked_data)
            loss_G = criterion_G(fake_data, real_data)
            loss_G.backward()
            optG.step()

            print('PRETRAIN [%d/%d][%d/%d] Loss_G: %.4f'
                  % (epoch + 1, opt.niter, i + 1, n_batches, loss_G.item()))

            lossG_all.update(loss_G.item())

        # Train D and G together:
        else:
            optD.zero_grad()

            # Feed the discriminator with real data:
            out = netD(real_data)
            lossD_real = criterion(out, real_label)

            # Feed the discriminator with fake data:
            fake_data = netG(masked_data)
            out = netD(fake_data.detach())
            lossD_fake = criterion(out, fake_label)

            # Update the loss
            loss_D = (lossD_real + lossD_fake) * opt.alpha
            loss_D.backward()
            optD.step()

            # Train the generator
            # Add noise:
            optG.zero_grad()
            masks = cuboid_noise(opt)
            masks = torch.from_numpy(masks)
            masks = masks.type(torch.FloatTensor).cuda()
            masked_data = real_data + masks
            masked_data[masked_data > 1] = 1

            fake_data = netG(masked_data)
            out = netD(fake_data)
            loss_G = criterion(out, real_label)
            loss_G.backward()
            optG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                  % (epoch + 1, opt.niter, i + 1, n_batches, loss_D, loss_G))

            lossD_all.update(loss_D.item())
            lossG_all.update(loss_G.item())

        # visdom visualization
        #if (i % 10) == 0 and i > 0:
        #    # vis.close(None)
        #    id_ch = np.random.randint(0, opt.batch_size, opt.batch_size)
        #    t = fake_data.detach().cpu().clone()
        #    t = t.permute(0, 4, 2, 3, 1)
        #    gen_data_np = t.numpy()
        #    t = masked_data.detach().cpu().clone()
        #    t = t.permute(0, 4, 2, 3, 1)
        #    masked_data_np = t.numpy()
        #    t = real_data.detach().cpu().clone()
        #    t = t.permute(0, 4, 2, 3, 1)
        #    real_data_np = t.numpy()
        #    for j in range(opt.batchSize):
        #        if gen_data_np[id_ch[j]].max() > 0.5:
        #            plotVoxelVisdom(np.squeeze(real_data_np[id_ch[j]] > 0.5), vis, '_'.join(map(str, [epoch, j])))
        #            plotVoxelVisdom(np.squeeze(masked_data_np[id_ch[j]] > 0.5), vis,
        #                            '_'.join(map(str, [epoch, j + 1])))
        #            plotVoxelVisdom(np.squeeze(gen_data_np[id_ch[j]] > 0.5), vis,
        #                            '_'.join(map(str, [epoch, j + 2])))
        #
        #            break

    print('Time elapsed Epoch %d: %d seconds'
          % (epoch + 1, time.time() - time0))

    # Dict update:
    loss_dict['D loss'].append(lossD_all.avg)
    loss_dict['G loss'].append(lossG_all.avg)

    # TensorBoard logging
    # scalar values
    #info = {
    #    'D loss': lossD_all.avg,
    #    'G loss': lossG_all.avg
    #}

    # for tag, value in info.items():
    #    logger.scalar_summary(tag, value, epoch)

    # values and gradients of the parameters (histogram)
    # for tag, value in netG.named_parameters():
    #    tag = tag.replace('.', '/')
    #    logger.histo_summary(tag, value.cpu().detach().numpy(), epoch)

    iteration = str(optG.state_dict()['state'][optG.state_dict()['param_groups'][0]['params'][0]]['step'])

    if (epoch % 5) == 0:
        torch.save(fake_data, GEN_DATA_DIR + '/netG_e%d.pt' % (epoch + 1))
        # print("gen_data:{} \n".format(gen_data))
        # t = gen_data.detach().cpu().clone()
        # t = t.permute(0, 4, 2, 3, 1)
        # print("gen_data_np:{} \n".format(gen_data_np))
        samples = fake_data.cpu().data[:8].squeeze().numpy()
        SavePloat_Voxels(samples, GEN_DATA_DIR, iteration)
        save_checkpoint({
          'epoch': epoch + 1,
          'state_dict': netG.state_dict(),
          'optimizer': optG.state_dict(),
          }, epoch + 1)



# ---------------------------------------------------
# Visualization:
# ---------------------------------------------------

import matplotlib.pyplot as plt


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(24, 8))
ax1.plot(range(1, len(loss_dict['D loss'])+1), loss_dict['D loss'], label='Training Loss of Discriminator')
ax1.set_title('Training Loss of Discriminator')
ax1.set(xlabel='Epochs', ylabel= 'Loss')
ax2.plot(range(1, len(loss_dict['G loss'])+1), loss_dict['G loss'], label='Training Loss of Generator')
ax2.set_title('Training Loss of Generator')
ax2.set(xlabel='Epochs', ylabel= 'Loss')
# plt.show()

plt.savefig(GEN_DATA_DIR + '/{}_leaky_soft_label_3DGAN.png'.format(obj))

print("The final epoch loss of Discriminator: {}, and Generator: {}".format(loss_dict['D loss'][-1],loss_dict['G loss'][-1]))