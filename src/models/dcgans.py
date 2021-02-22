import torch.nn as nn
import torch
from collections import OrderedDict
import logging
import math

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
      nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
      nn.init.normal_(m.weight.data, 1.0, 0.02)
      nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
  
  def __init__(self, size, nz, ngf, nc):
    """ Initialization 
    Arguments:
    size: int - Size of the resulting image.
    nz: int - Length of noise vector.
    ngf: int - Relates to number of features.  TODO: right now only 64 will work
    nc: int - Number of channels.
    """
    super(Generator, self).__init__()

    newsize = int(pow(2,math.floor(math.log(size,2))))
    logging.info("Size specified: " + str(size) +
                 " Size used: " + str(newsize))
    self.size = newsize

    newngf = int(pow(2,math.floor(math.log(ngf,2))))
    logging.info("ngf specified: " + str(ngf) +
                 " ngf used: " + str(newngf))
    ngf = newngf

    layers = []
    layer_num = 0

    mult = int(self.size / 8)

    # Add the initial layer
    layers.append((str(layer_num) + "_ConvTranspose2d",
      nn.ConvTranspose2d( nz, ngf * mult, 4, 1, 0, bias=False)))
    layers.append((str(layer_num) + "_BatchNorm2d",
      nn.BatchNorm2d(ngf * mult)))
    layers.append((str(layer_num) + "_ReLU",
      nn.ReLU(True)))

    while mult > 1:
      mult = mult / 2
      imult = int(mult)
      layer_num += 1
      
      layers.append((str(layer_num) + "_ConvTranspose2d",
        nn.ConvTranspose2d( ngf * 2 * imult, ngf * imult, 4, 2, 1, bias=False)))
      layers.append((str(layer_num) + "_BatchNorm2d",
        nn.BatchNorm2d(ngf * imult)))
      layers.append((str(layer_num) + "_ReLU",
        nn.ReLU(True)))


    layer_num += 1
    layers.append((str(layer_num) + "_ConvTranspose2d",
      nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False)))
    layers.append((str(layer_num) + "_Tanh",
      nn.Tanh()))

    d = OrderedDict(layers)

    self.model = nn.Sequential(d)
   
    
  def forward(self, input):
    return self.model(input)


class Discriminator(nn.Module):

  def __init__(self, size, ndf, nc):
    """ Initialization 
    Arguments:
    size: int - Size of the resulting image.
    ndf: int - Relates to number of features.  TODO: right now only 64 will work
    nc: int - Number of channels.
    """
    super(Discriminator, self).__init__()
   
    newsize = int(pow(2,math.floor(math.log(size,2))))
    logging.info("Size specified: " + str(size) +
                 " Size used: " + str(newsize))
    self.size = newsize

    newndf = int(pow(2,math.floor(math.log(ndf,2))))
    logging.info("ndf specified: " + str(ndf) +
                 " ndf used: " + str(newndf))
    ndf = newndf

    layers = []
    layer_num = 0

    mult = 1/2 

    # Add the initial layer
    layers.append((str(layer_num) + "_Conv2d",
      nn.Conv2d( nc, ndf, 4, 2, 1, bias=False)))
    layers.append((str(layer_num) + "_LeakyReLU",
      nn.LeakyReLU(True)))

    while mult < int(self.size / 16):
      mult = mult * 2
      imult = int(mult)
      layer_num += 1
      
      layers.append((str(layer_num) + "_Conv2d",
        nn.Conv2d( ndf * imult, ndf * 2 * imult, 4, 2, 1, bias=False)))
      layers.append((str(layer_num) + "_BatchNorm2d",
        nn.BatchNorm2d(ndf * 2 * imult)))
      layers.append((str(layer_num) + "_LeakyReLU",
        nn.LeakyReLU(True)))


    mult = mult * 2
    layer_num += 1
    layers.append((str(layer_num) + "_Conv2d",
      nn.Conv2d( int(mult) * ndf, 1, 4, 1, 0, bias=False)))
    layers.append((str(layer_num) + "_Sigmoid",
      nn.Sigmoid()))

    d = OrderedDict(layers)

    self.model = nn.Sequential(d)


  def forward(self, input):
    return self.model(input) 
    
