# This executes the GAN and evaluates it
# problem ids
# 1-15: GAN from level 1-15 with direct fitness
# 16-30: GAN from level 1-15 with sim fitness
# Available dimensions 10, 20, 30, 40
import torch
import torchvision.utils as vutils
from torch.autograd import Variable

import sys
import os
import json
import numpy
import pytorch.models.dcgan as dcgan
import random
import math
import matplotlib.pyplot as plt

batchSize = 64

imageSize = 32
ngf = 64
ngpu = 1
n_extra_layers = 0

features = 10
budget=50

GROUND = 0
ENEMY = 5
PIPE = 6 #7, 8 9

batchSize = 1

def count_tile_type(x, nz, tile):
    x = numpy.array(x)
    latent_vector = torch.FloatTensor(x).view(batchSize, nz, 1, 1)
    levels = generator(Variable(latent_vector, volatile=True))
    levels.data = levels.data[:, :, :14, :28]
    im = levels.data.cpu().numpy()
    im = numpy.argmax( im, axis = 1)
    num_tiles = (len (im[im==tile]))
    return num_tiles

def gan_maximse_tile_type(x, nz, tile):
    return -count_tile_type(x, nz, tile)

def gan_target_tile_type(x, nz, tile, target):
    return abs(target-count_tile_type(x,nz, tile))

def gan_target_tile_type_frac(x, nz, tile, target_frac):
    total_tiles = 14*28
    if tile==GROUND:
        total_tiles=28
    return abs(target_frac-(count_tile_type(x,nz, tile)/(total_tiles)))


def fitnessSO(x, nz):
    return gan_target_tile_type_frac(x,nz, GROUND, 0.8)*gan_target_tile_type(x, nz, ENEMY, 5)



#expecting variables <prob> <dim>
if __name__ == '__main__':
    _, dim, problem = sys.argv
    problem = int(problem)
    dim = int(dim)

    available_dims = [10,20,30,40]

    if dim not in available_dims: #check Dimension available
        raise ValueError("asked for dimension '{}', but is not available".format(dim))

    # Read the variables
    with open('variables.txt') as f:
        content = f.readlines()
        content = [float(line.rstrip('\n')) for line in content]
        num_variables = content[0]
        if num_variables != dim: #check appropriate number of variables there
            raise ValueError("num_variables should be '{}', but is '{}'"
                             "".format(dim, num_variables))
        f.close()

    #check variables in range
    inp = numpy.array(content[1:])
    if numpy.any(inp>1) or  numpy.any(inp<-1):#input out of range
        with open('objectives.txt', 'w') as f: #write out NaN result
            f.write('{}\n'.format(0))
            f.close()
    else:
        netG = "GAN/samples-{0}-{1}-{2}/netG_epoch_{0}_{1}_{2}.pth".format(budget, problem%15, dim)
    
        # Compute the result
        if problem<=15: #direct evaluation
            generator = dcgan.DCGAN_G(imageSize, dim, features, ngf, ngpu, n_extra_layers)
            generator.load_state_dict(torch.load(netG, map_location=lambda storage, loc: storage))
            result = fitnessSO(content[1:], dim)
            #print(count_tile_type(content[1:], dim, GROUND))
            #print(count_tile_type(content[1:], dim, ENEMY))
            #print(count_tile_type(content[1:], dim, PIPE))

            # Write the result
            with open('objectives.txt', 'w') as f:
                f.write('{}\n'.format(1))
                f.write('{}\n'.format(result))
                f.close()
        else:
            os.system('java -jar marioaiDagstuhl.jar "'+str(content[1:])+'" '+netG+' '+str(dim))
