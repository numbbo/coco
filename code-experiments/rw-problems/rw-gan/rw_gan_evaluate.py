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

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:]
    image = numpy.zeros((height*shape[0], width*shape[1],shape[2]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img
    return image


batchSize = 1

def gan_maximse_title_type(x):
    x = numpy.array(x)
    latent_vector = torch.FloatTensor(x).view(batchSize, nz, 1,
                                              1)  # torch.from_numpy(lv)# torch.FloatTensor( torch.from_numpy(lv) )
    levels = generator(Variable(latent_vector, volatile=True))
    levels.data = levels.data[:, :, :14, :28]
    im = levels.data.cpu().numpy()
    im = numpy.argmax( im, axis = 1)

    num_titles =  (len (im[im == PIPE]))
    return 100.0 - num_titles


def gan_fitness_function(x, nz):
    x = numpy.array(x)
    # print(x)

    latent_vector = torch.FloatTensor(x).view(batchSize, nz, 1,
                                              1)  # torch.from_numpy(lv)# torch.FloatTensor( torch.from_numpy(lv) )
    levels = generator(Variable(latent_vector, volatile=True))
    levels.data = levels.data[:, :, :14, :28]
    #return solid_blocks_fraction(levels.data, 0.2)
    return solid_blocks_fraction(levels.data, 0.4)*ground_blocks_fraction(levels.data,0.8)


def ground_blocks_fraction(data, frac):
    ground_count = sum(data[0, GROUND, 13, :] > 0)
    #print(ground_count)
    #print(ground_count- frac*28)
    return math.sqrt(math.pow(ground_count - frac*28, 2))

def solid_blocks_fraction(data, frac):
    solid_block_count = len(data[data > 0.])
    return math.sqrt(math.pow(solid_block_count - frac*14*28, 2))

#expecting variables <prob> <dim>
if __name__ == '__main__':
    _, problem, dim = sys.argv
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
            result = gan_fitness_function(content[1:], dim)

            # Write the result
            with open('objectives.txt', 'w') as f:
                f.write('{}\n'.format(1))
                f.write('{}\n'.format(result))
                f.close()
        else:
            os.system('java -jar marioaiDagstuhl.jar "'+str(content[1:])+'" '+netG+' '+str(dim))
