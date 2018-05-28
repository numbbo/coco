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
BREAK = 1
PASS = 2
QUESTION = 3
EQUESTION = 4
ENEMY = 5
LEFTTOPPIPE = 6
RIGHTTOPPIPE = 7
LEFTBOTPIPE = 8
RIGHTBOTPIPE = 9
COIN = 10

#        "X" : ["solid","ground"],
#        "S" : ["solid","breakable"],
#        "-" : ["passable","empty"],
#        "?" : ["solid","question block", "full question block"],
#        "Q" : ["solid","question block", "empty question block"],
#        "E" : ["enemy","damaging","hazard","moving"],
#        "<" : ["solid","top-left pipe","pipe"],
#        ">" : ["solid","top-right pipe","pipe"],
#        "[" : ["solid","left pipe","pipe"],
#        "]" : ["solid","right pipe","pipe"],
#	     "o" : ["coin","collectable","passable"]


batchSize = 1

def exist_gap(im):
    #gap exists if not 10 (coin), not 2 (passable)
    gaps = numpy.zeros(28)
    for i in range(0,28):
        imc = im[:,:,i][0]
        unique, counts = numpy.unique(imc, return_counts=True)
        dist = dict(zip(unique, counts))
        if dist.get(COIN,0) + dist.get(PASS,0)==14:#all tiles in column passable
            gaps[i]=1
    return gaps

def count_gaps(im):
    gaps = exist_gap(im)
    return sum(gaps)

def gap_lengths(im):
    gaps = exist_gap(im)
    gaps = "".join([str(int(x)) for x in gaps])
    return map(len,gaps.split('0'))

def max_gap(im):
    return max(gap_lengths(im))


def count_tile_type(im, tile):
    num_tiles = (len (im[im==tile]))
    return num_tiles

def gan_maximse_tile_type(x, nz, tile):
    return -count_tile_type(x, nz, tile)

def gan_target_tile_type(x, nz, tile, target):
    return abs(target-count_tile_type(x,nz, tile))

def gan_target_tile_type_frac(im, tile, target_frac):
    total_tiles = 14*28
    return abs(target_frac-(count_tile_type(im, tile)/(total_tiles)))

def leniency(im):
    unique, counts = numpy.unique(im, return_counts=True)
    dist = dict(zip(unique, counts))
    val =0
    val += dist.get(QUESTION,0)*1
    val += dist.get(ENEMY,0)*(-1)
    val += count_gaps(im)*(-0.5)
    t = numpy.array(gap_lengths(im))
    val -= numpy.mean(t[t!=0])
    print(dist)
    return val

def fitnessSO(x, nz):
    x = numpy.array(x)
    latent_vector = torch.FloatTensor(x).view(batchSize, nz, 1, 1)
    levels = generator(Variable(latent_vector, volatile=True))
    levels.data = levels.data[:, :, :14, :28]
    im = levels.data.cpu().numpy()
    im = numpy.argmax( im, axis = 1)
    print(im)
    #return count_gaps(im)
    #return max_gap(im)
    return leniency(im)
    #return gan_target_tile_type_frac(im, GROUND, 0.8)*gan_target_tile_type(im, ENEMY, 5)



#expecting variables <prob> <dim> <instance>
if __name__ == '__main__':
    _, dim, problem, instance = sys.argv
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
            print(result)
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
