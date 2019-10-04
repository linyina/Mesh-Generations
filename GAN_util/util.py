# original code source https://github.com/meetshah1995/tf-3dgan/blob/master/src/dataIO.py

import os
import scipy.ndimage as nd
import scipy.io as io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import skimage.measure as sk
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import numpy as np
import pickle

try:
    import trimesh
    from stl import mesh
except:
    pass
    print('All dependencies not loaded, some functionality may not work')




# DATA DIR:
DATA_DIR = '/home/yina_lin_18/3DMeshConvGan/data/'
MODEL_DIR = '/home/yina_lin_18/3DMeshConvGan/3DGAN/models/'
LOG_DIR = '/home/yina_lin_18/3DMeshConvGan/3DGAN/logs/'
GEN_DATA_DIR = '/home/yina_lin_18/3DMeshConvGan/3DGAN/gen_data/'


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


def cuboid_noise(opt):
    mask = []
    for i in range(opt.batch_size):
        x1, y1, z1 = np.random.randint(0, opt.imageSize - opt.localSize + 1, 3)

        cube = np.zeros((1, opt.imageSize, opt.imageSize, opt.imageSize), dtype=np.uint8)

        w, h, d = np.random.randint(opt.hole_min, opt.hole_max + 1, 3)
        p1 = x1 + np.random.randint(0, opt.localSize - w)
        q1 = y1 + np.random.randint(0, opt.localSize - h)
        r1 = z1 + np.random.randint(0, opt.localSize - d)
        p2 = p1 + w
        q2 = q1 + h
        r2 = r1 + d

        cube[:, q1:q2 + 1, p1:p2 + 1, r1:r2 + 1] = 1
        mask.append(cube)

    return np.array(mask)


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

