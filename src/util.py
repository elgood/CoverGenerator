from torchvision.utils import save_image
import torchvision.transforms.functional as F
import torch
import numpy as np
import os

class SquarePad:
  def __call__(self, image):
    w, h = image.size
    max_wh = np.max([w, h])
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)
    padding = (hp, vp, hp, vp)
    return F.pad(image, padding, 0, 'constant')


def write_images(epoch, path, fixed_noise, num_test_samples, 
                 netG, device, use_fixed=False):
  z = torch.randn(num_test_samples, 100, 1, 1, device=device)
  if use_fixed:
    with torch.no_grad():
      generated_fake_images = netG(fixed_noise)
  else:
    with torch.no_grad():
      generated_fake_images = netG(z)

  for i in range(len(generated_fake_images)):
    filename = str(epoch) + "_" + str(i) + ".png"
    filepath = os.path.join(path, filename)
    save_image(generated_fake_images[i], filepath)


