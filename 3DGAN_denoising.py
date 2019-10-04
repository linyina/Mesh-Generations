# ALL IN ONE: -> more convenient to run in Colab


# ---------------------------------------------------
# Import Functions
# ---------------------------------------------------

from __future__ import print_function
# import argparse
import os
import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import skimage.measure as sk
from stl import mesh
import sys
import os

import scipy.ndimage as nd
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt


from mpl_toolkits import mplot3d

try:
    import trimesh
    from stl import mesh
except:
    pass
    print('All dependencies not loaded, some functionality may not work')

import tensorflow as tf
import numpy as np
import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

import scipy.ndimage as nd
import scipy.io as io
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.measure as sk
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils import data
from torch.autograd import Variable
import torch
import os
import pickle

# ---------------------------------------------------
# DIR:
# ---------------------------------------------------

DATA_DIR = '/home/yina_lin_18/3DMeshConvGan/data/'
MODEL_DIR = '/home/yina_lin_18/3DMeshConvGan/3DGAN/models/'
LOG_DIR = '/home/yina_lin_18/3DMeshConvGan/3DGAN/logs/'
fake_data_DIR = '/home/yina_lin_18/3DMeshConvGan/3DGAN/fake_data/'


# ---------------------------------------------------
# UTIL FUNCTIONS:
# ---------------------------------------------------
class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


# ---------------------------------------------------
# Utils:
# ---------------------------------------------------

def getVF(path):
    raw_data = tuple(open(path, 'r'))
    header = raw_data[1].split()
    n_vertices = int(header[0])
    n_faces = int(header[1])
    vertices = np.asarray([map(float,raw_data[i+2].split()) for i in range(n_vertices)])
    faces = np.asarray([map(int,raw_data[i+2+n_vertices].split()) for i in range(n_faces)])
    return vertices, faces

def plotFromVF(vertices, faces):
    input_vec = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            input_vec.vectors[i][j] = vertices[f[j],:]
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(input_vec.vectors))
    scale = input_vec.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)
    plt.show()

def plotFromVoxels(voxels):
    z,x,y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c= 'red')
    plt.show()

def getVFByMarchingCubes(voxels, threshold=0.5):
    v, f = sk.marching_cubes_classic(voxels, level=threshold)
    return v, f

def plotMeshFromVoxels(voxels, threshold=0.5):
    v,f = getVFByMarchingCubes(voxels, threshold)
    plotFromVF(v,f)

def plotVoxelVisdom(voxels, visdom, title):
    v, f = getVFByMarchingCubes(voxels)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))

def plotFromVertices(vertices):
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)
    axes.scatter(vertices.T[0,:],vertices.T[1,:],vertices.T[2,:])
    plt.show()

def getVolumeFromOFF(path, sideLen=32):
    mesh = trimesh.load(path)
    volume = trimesh.voxel.Voxel(mesh, 0.5).raw
    (x, y, z) = map(float, volume.shape)
    volume = nd.zoom(volume.astype(float),
                     (sideLen/x, sideLen/y, sideLen/z),
                     order=1,
                     mode='nearest')
    volume[np.nonzero(volume)] = 1.0
    return volume.astype(np.bool)

def getVoxelFromMat(path, cube_len=64):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels,(1,1),'constant',constant_values=(0,0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2,2,2), mode='constant', order=0)
    return voxels

def getAll(obj='chair',train=True, cube_len=64, obj_ratio=1.0):
    # objPath = DATA_DIR + obj
    objPath = DATA_DIR + obj + '/30/'
    objPath += 'train/' if train else 'test/'
    fileList = [f for f in os.listdir(objPath) if f.endswith('.mat')]
    # fileList = [f for f in os.listdir(objPath) if f.endswith('.off')]
    fileList = fileList[0:int(obj_ratio*len(fileList))]
    volumeBatch = np.asarray([getVoxelFromMat(objPath + f, cube_len) for f in fileList],dtype=np.bool)
    # volumeBatch = np.asarray([getVolumeFromOFF(objPath + f, side_len) for f in fileList], dtype=np.bool)
    return volumeBatch


def SavePloat_Voxels(voxels, path, iteration):
    voxels = voxels[:8].__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.set_aspect('equal')
    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()

    with open(path + '/{}.pkl'.format(str(iteration).zfill(3)), "wb") as f:
        pickle.dump(voxels, f, protocol=pickle.HIGHEST_PROTOCOL)

# ---------------------------------------------------
# OPTIONS:
# ---------------------------------------------------
import easydict
opt = easydict.EasyDict({
    "obj":'piano',
    "workers": 2,
    "cuda": 1,
    "batchSize": 64,
    "imageSize":64,
    "localSize":32,
    "hole_min":12,
    "hole_max":18,
    "ndf":64,
    "niter":300,
    "preniter":20,
    "lrG":0.0002,
    "lrD":0.0001,
    "alpha":0.01,
    "beta1":0.5,
    "beta2": 0.99,
    "manualSeed":None,
    "LeakyReLu": True,
    "leak_value": 0.2,
    "bias": False,
    "G_output_activation":nn.Tanh(),
    "soft_label": True
})


print(opt)

# ---------------------------------------------------
# SETUPS:
# ---------------------------------------------------

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
obj = 'table'
obj_ratio = 0.7


if not os.path.exists('logs'):
  os.makedirs('logs')

# save feedback
# logger = Logger(LOG_DIR)

# init visdom server for visualization
# vis = visdom.Visdom()

#create a directory to save the trained model
if not os.path.exists('models'):
  os.makedirs('models')


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Generator(nn.Module):
    def __init__(self,opt):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Conv3d
            nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1, bias=opt.bias),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(opt.leak_value),

            # Conv3d
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1, bias=opt.bias),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(opt.leak_value),

            # Conv3d
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1, bias=opt.bias),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(opt.leak_value),

            # Conv3d
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=opt.bias),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(opt.leak_value),

            # Dilated conv3d
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4, bias=opt.bias),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(opt.leak_value),

            # Dilated conv3d
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=8, dilation=8, bias=opt.bias),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(opt.leak_value),

            # Conv3d
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=opt.bias),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(opt.leak_value),

            # Deconv3d
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1, bias=opt.bias),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(opt.leak_value),

            # Deconv3d
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1, bias=opt.bias),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(opt.leak_value),

            # Deconv3d
            nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1, bias=opt.bias),
            opt.G_output_activation
        )

    def forward(self, input):
        output = self.main(input)

        return output

class Discriminator(nn.Module):
    def __init__(self,opt):
        super(Discriminator, self).__init__()
        self.encoder = nn.Sequential(
            # Conv3d
            nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1, bias=opt.bias),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(opt.leak_value),

            # Conv3d
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1, bias=opt.bias),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(opt.leak_value),

            # Conv3d
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1, bias=opt.bias),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(opt.leak_value),

            # Conv3d
            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1, bias=opt.bias),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(opt.leak_value)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x.view(-1, 1).squeeze(1)

def weights_init(m):
    for m in m.modules():
        if isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2./n))
        elif isinstance(m, nn.ConvTranspose3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.normal_(0, 1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

def get_points():
    points = []
    mask = []
    for i in range(opt.batchSize):
        x1, y1, z1 = np.random.randint(0, opt.imageSize - opt.localSize + 1, 3)
        x2, y2, z2 = np.array([x1, y1, z1]) + opt.localSize
        points.append([x1, y1, x2, y2, z1, z2])

        w, h, d = np.random.randint(opt.hole_min, opt.hole_max + 1, 3)
        p1 = x1 + np.random.randint(0, opt.localSize - w)
        q1 = y1 + np.random.randint(0, opt.localSize - h)
        r1 = z1 + np.random.randint(0, opt.localSize - d)
        p2 = p1 + w
        q2 = q1 + h
        r2 = r1 + d

        m = np.zeros((1, opt.imageSize, opt.imageSize, opt.imageSize), dtype=np.uint8)
        m[:, q1:q2 + 1, p1:p2 + 1, r1:r2 + 1] = 1
        mask.append(m)

    return np.array(points), np.array(mask)

def save_checkpoint(state, obj, curr_epoch):
    torch.save(state, MODEL_DIR + obj + 'netG_e%d.pth.tar' % (curr_epoch))


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
# load ".mat" files
volumes = getAll(obj=obj, train=True, obj_ratio=obj_ratio)
print('Using ' + obj + ' Data')
volumes = volumes[..., np.newaxis].astype(np.float)


data = torch.from_numpy(volumes)
data = data.permute(0, 4, 1, 2, 3)
data = data.type(torch.FloatTensor)



# ---------------------------------------------------
# Define Loss criterion:
# ---------------------------------------------------
# choose loss function
criterion_D = nn.BCELoss()
criterion_G = nn.MSELoss()


# ---------------------------------------------------
# Optimizers:
# ---------------------------------------------------
optG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
optD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))



# ---------------------------------------------------
# Setup Labels:
# ---------------------------------------------------

if opt.soft_label:
    real_label = Variable(torch.Tensor(opt.batchSize).uniform_(0.7, 1.2).cuda())
    fake_label = Variable(torch.Tensor(opt.batchSize).uniform_(0, 0.3).cuda())
else:
    real_label = torch.full((opt.batchSize,), 1, device=device)
    fake_label = torch.full((opt.batchSize,), 0, device=device)


# ---------------------------------------------------
# Setup Loss storages:
# ---------------------------------------------------

errD_all = AverageMeter()
errG_all = AverageMeter()

loss_dict = {
    'D loss':[],
    'G loss':[]
}

# ---------------------------------------------------
# Training:
# ---------------------------------------------------
n_batches = int(data.size(0) / opt.batchSize)
for epoch in range(opt.niter):
    time0 = time.time()
    data_perm = torch.randperm(data.size(0))

    for i in range(n_batches):

        # Create cuboid noise
        points_batch, mask_batch = get_points()

        batch = data[i * opt.batchSize:(i * opt.batchSize + opt.batchSize)]
        real_data = batch.to(device)

        # Adding noise to the batch data
        temp = torch.from_numpy(mask_batch)
        masks = temp.type(torch.FloatTensor).cuda()
        masked_data = real_data + masks
        masked_data[masked_data > 1] = 1

        # Warm Up
        if epoch < opt.preniter:
            optG.zero_grad()
            fake_data = netG(masked_data)
            errG = criterion_G(fake_data, real_data)
            errG.backward()
            optG.step()

            print('PRETRAIN [%d/%d][%d/%d] Loss_G: %.4f'
                  % (epoch + 1, opt.niter, i + 1, n_batches, errG.item()))

            errG_all.update(errG.item())

        # Adversarial Traning
        else:
            # Train the generator five times for each training on discriminator
            if (epoch % 5) == 0:
                optD.zero_grad()

                out = netD(real_data)
                errD_real = criterion_D(out, real_label)

                # train Discriminator with generated samples

                fake_data = netG(masked_data)
                out = netD(fake_data.detach())
                errD_fake = criterion_D(out, fake_label)
                errD = (errD_real + errD_fake) * opt.alpha   # To reduce the learning rate of discriminator
                errD.backward()
                optD.step()

            # update Generator
            optG.zero_grad()
            fake_data = netG(masked_data)
            out = netD(fake_data)
            errG = criterion_D(out, real_label)
            errG.backward()
            optG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                  % (epoch + 1, opt.niter, i + 1, n_batches, errD, errG))

            errD_all.update(errD.item())
            errG_all.update(errG.item())

        # visdom visualization
        #if (i % 30) == 0 and i > 0:
        #    # vis.close(None)
        #    id_ch = np.random.randint(0, opt.batchSize, opt.batchSize)
        #    t = fake_data.detach().cpu().clone()
        #    t = t.permute(0, 4, 2, 3, 1)
        #    fake_data_np = t.numpy()
        #    t = masked_data.detach().cpu().clone()
        #    t = t.permute(0, 4, 2, 3, 1)
        #    masked_data_np = t.numpy()
        #    t = real_data.detach().cpu().clone()
        #    t = t.permute(0, 4, 2, 3, 1)
        #    real_data_np = t.numpy()
        #    for j in range(opt.batchSize):
        #        if fake_data_np[id_ch[j]].max() > 0.3:
        #            plotVoxelVisdom(np.squeeze(real_data_np[id_ch[j]] > 0.3), vis, '_'.join(map(str, [epoch, j])))
        #            plotVoxelVisdom(np.squeeze(masked_data_np[id_ch[j]] > 0.3), vis, '_'.join(map(str, [epoch, j + 1])))
        #            plotVoxelVisdom(np.squeeze(fake_data_np[id_ch[j]] > 0.3), vis, '_'.join(map(str, [epoch, j + 2])))
        #            break

    print('Time elapsed Epoch %d: %d seconds'
          % (epoch + 1, time.time() - time0))

    # Dict update:
    loss_dict['D loss'].append(errD_all.avg)
    loss_dict['G loss'].append(errG_all.avg)

    # TensorBoard logging
    # scalar values
    #info = {
    #    'D loss': errD_all.avg,
    #    'G loss': errG_all.avg
    #}

    # for tag, value in info.items():
    #    logger.scalar_summary(tag, value, epoch)

    # values and gradients of the parameters (histogram)
    # for tag, value in netG.named_parameters():
    #    tag = tag.replace('.', '/')
    #    logger.histo_summary(tag, value.cpu().detach().numpy(), epoch)

    iteration = str(optG.state_dict()['state'][optG.state_dict()['param_groups'][0]['params'][0]]['step'])
    if (epoch % 5) == 0:
        torch.save(fake_data, fake_data_DIR + obj + '/netG_e%d.pt' % (epoch + 1))
        # print("fake_data:{} \n".format(fake_data))
        # t = fake_data.detach().cpu().clone()
        # t = t.permute(0, 4, 2, 3, 1)
        # print("fake_data_np:{} \n".format(fake_data_np))
        samples = fake_data.cpu().data[:8].squeeze().numpy()
        SavePloat_Voxels(samples, fake_data_DIR + obj , iteration)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': netG.state_dict(),
            'optimizer': optG.state_dict(),
        }, obj, epoch + 1)



# ---------------------------------------------------
# Loss Visualization:
# ---------------------------------------------------
import matplotlib.pyplot as plt


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(24, 8))
ax1.plot(range(1, len(loss_dict['D loss'][21:])+1), loss_dict['D loss'][21:], label='Training Loss of Discriminator')
ax1.set_title('Training Loss of Discriminator')
ax1.set(xlabel='Epochs', ylabel= 'Loss')
ax2.plot(range(1, len(loss_dict['G loss'])+1), loss_dict['G loss'], label='Training Loss of Generator')
ax2.set_title('Training Loss of Generator')
ax2.set(xlabel='Epochs', ylabel= 'Loss')


# plt.show()

plt.savefig('table_final_3DGAN.png')

print("The final epoch loss of Discriminator: {}, and Generator: {}".format(loss_dict['D loss'][-1],loss_dict['G loss'][-1]))



# ---------------------------------------------------
# Test:
# ---------------------------------------------------

volumes_test = getAll(obj=obj, train=False, obj_ratio=obj_ratio)
print('Using ' + obj + ' Data')
volumes_test = volumes_test[..., np.newaxis].astype(np.float)

test_data = torch.from_numpy(volumes_test)
test_data = test_data.permute(0, 4, 1, 2, 3)
test_data = test_data.type(torch.FloatTensor)


test_loss = []
for i in range(10):

    points_batch, mask_batch = get_points()
    test_batch = test_data[i:i*opt.batchSize]
    test_real_data = test_batch.to(device)
    # add noise to batch
    temp = torch.from_numpy(mask_batch)
    masks = temp.type(torch.FloatTensor).cuda()

    masked_data = test_real_data + masks
    masked_data[masked_data > 1] = 1

    test_gen_data = netG(masked_data)
    Generalisation_error = torch.mean((test_gen_data - test_real_data) ** 2)
    test_loss.append(Generalisation_error)


test_loss = np.array(test_loss)
print("The mean generalisation error is: {}. \n The std of test loss is: {}.".format(np.mean(test_loss), np.std(test_loss)))