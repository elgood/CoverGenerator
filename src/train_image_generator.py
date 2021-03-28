import argparse
import args
import logging
import torch
import torchvision
from torchvision import transforms
import os
import sys
from time import time

from models import dcgans
import util

# Developed from tutorial:
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

def main():

  parser = args.get_default_ArgumentParser(
    "Trains a GANs that will generate artificial images.")
  
  parser.add_argument("images", type=str,
    help="Path to folder containing images.  Note, the folder must have" +
      " subdirectories of images.")
  parser.add_argument("--model", choices=["dcgans"], default="dcgans",
    help="The type of model to train.")
  parser.add_argument("--size", type=int, choices=[64, 128, 256, 512], 
    default=64, help="The size of the output image.")
  parser.add_argument("--nz", type=int, default=100,
    help="Size of latent noise vector.")
  parser.add_argument("--ngf", type=int, choices=[64], default=64,
    help="Size of feature maps in generator.")
  parser.add_argument("--ndf", type=int, choices=[64], default=64,
    help="Size of feature maps in discriminator.")
  parser.add_argument("--nc", type=int, choices=[3], default=3,
    help="Size of feature maps in generator.")
  parser.add_argument("--gpus", nargs='+', type=int,
    help="List of gpu ids to use.")
  parser.add_argument("--num_workers", type=int, default=2,
    help="Number of dataloader workers.")
  parser.add_argument("--batch_size", type=int, default=128,
    help="Batch size during training.") 
  parser.add_argument("--dlr", type=float, default=0.0002,
    help="Discriminator learning rate.")
  parser.add_argument("--glr", type=float, default=0.0002,
    help="Generator learning rate.")
  parser.add_argument("--num_epochs", type=int, default=100,
    help="Number of epochs.")
  parser.add_argument("--beta1", type=float, default=0.5,
    help="Adam optimizer parameter.")
  parser.add_argument("--checkpoint", type=str, default="checkpoints",
    help="Directory for checkpoints.")  
  parser.add_argument("--recover", type=str, default=None,
    help="Checkpoint to recover from.")
  parser.add_argument("--results", type=str, default="results",
    help="Place to put resulting images.")
  parser.add_argument("--use_fixed", action='store_true',
    help="Will use fixed noise to generate sample images.")

  FLAGS = parser.parse_args()

  args.process_common_arguments(FLAGS)

  device = torch.device("cuda:0" if (torch.cuda.is_available() and 
            gpus is not None) else "cpu")

  logging.info("Using gpus: " + str(FLAGS.gpus))

  if FLAGS.model == "dcgans":
    netG = dcgans.Generator(FLAGS.size,FLAGS.nz, FLAGS.ngf, FLAGS.nc).to(device)
    netD = dcgans.Discriminator(FLAGS.size, FLAGS.ndf, FLAGS.nc).to(device)
    weights_init = dcgans.weights_init

  if (device.type == 'cuda') and (FLAGS.gpus is not None):
    generator = torch.nn.DataParallel(generator, FLAGS.gpus)
  if (device.type == 'cuda') and (FLAGS.gpus is not None):
    discriminator = torch.nn.DataParallel(discriminator, FLAGS.gpus)

  netG.apply(weights_init)
  netD.apply(weights_init)

  logging.info("Generator:")
  logging.info(netG)
  logging.info("Discriminator:")
  logging.info(netD)


  transform = transforms.Compose([
    util.SquarePad(),
    transforms.Resize(FLAGS.size),
    transforms.CenterCrop(FLAGS.size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  dataset = torchvision.datasets.ImageFolder(root=FLAGS.images, 
                                             transform=transform)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size,
                                    shuffle=True, num_workers=FLAGS.num_workers)
    
 
  criterion = torch.nn.BCELoss()

  fixed_noise = torch.randn(64, FLAGS.nz, 1, 1, device=device)

  real_label = 1.0
  fake_label = 0.0

  optimizerD = torch.optim.Adam(netD.parameters(), lr=FLAGS.dlr, 
    betas=(FLAGS.beta1, 0.999))
  optimizerG = torch.optim.Adam(netG.parameters(), lr=FLAGS.glr, 
    betas=(FLAGS.beta1, 0.999))
  
  G_loss = []
  D_loss = []
  beg_epoch = 0

  if FLAGS.recover is not None:
    if os.path.isfile(FLAGS.recover):
      logging.info("Recovering from checkpoint " + FLAGS.recover)
      checkpoint = torch.load(FLAGS.recover)
      beg_epoch = checkpoint['epoch']
      netG.load_state_dict(checkpoint['model_state_dictG'])
      optimizerG.load_state_dict(checkpoint['optimizer_state_dictG'])
      netD.load_state_dict(checkpoint['model_state_dictD'])
      optimizerD.load_state_dict(checkpoint['optimizer_state_dictD'])
      G_loss = checkpoint['lossG']
      D_loss = checkpoint['lossD']
      logging.info("Starting at epoch " + str(beg_epoch))

    else:
      print("Could not find specified checkpoint:", FLAGS.recover)
      sys.exit()

  
  logging.info("Starting training loop")
   
  # For each epoch
  for epoch in range(beg_epoch, FLAGS.num_epochs):
    t1 = time()
    # For each batch in the dataloader
    for i, data in enumerate(dataloader):

      ############################
      # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      ###########################
      ## Train with all-real batch
      netD.zero_grad()
      # Format batch
      #real_cpu = data[0].to(device)
      #b_size = real_cpu.size(0)
      real_images = data[0].to(device)
      b_size = real_images.shape[0]

      label = torch.full((b_size,), real_label, dtype=torch.float, 
                          device=device)
      # Forward pass real batch through D
      #output = netD(real_cpu).view(-1)
      output = netD(real_images).view(-1)
      # Calculate loss on all-real batch
      errD_real = criterion(output, label)
      # Calculate gradients for D in backward pass
      errD_real.backward()
      D_x = output.mean().item()

      ## Train with all-fake batch
      # Generate batch of latent vectors
      noise = torch.randn(b_size, FLAGS.nz, 1, 1, device=device)
      # Generate fake image batch with G
      fake = netG(noise)
      label.fill_(fake_label)
      # Classify all fake batch with D
      #output = netD(fake.detach()).view(-1)
      output = netD(fake.detach()).view(-1)
      # Calculate D's loss on the all-fake batch
      errD_fake = criterion(output, label)
      # Calculate the gradients for this batch
      errD_fake.backward()
      D_G_z1 = output.mean().item()
      # Add the gradients from the all-real and all-fake batches
      errD = errD_real + errD_fake
      # Update D
      optimizerD.step()
      print("errD", errD)

      ############################
      # (2) Update G network: maximize log(D(G(z)))
      ###########################
      netG.zero_grad()
      label.fill_(real_label)  # fake labels are real for generator cost
      # Since we just updated D, perform another forward pass of 
      # all-fake batch through D
      #output = netD(fake).view(-1)
      output = netD(fake).view(-1)
      # Calculate G's loss based on this output
      errG = criterion(output, label)
      # Calculate gradients for G
      errG.backward()
      D_G_z2 = output.mean().item()
      # Update G
      optimizerG.step()
      print("errG", errG)

      # Output training stats
      if i % 50 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
          % (epoch, FLAGS.num_epochs, i, len(dataloader),
          errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

      # Save Losses for plotting later
      G_loss.append(errG.item())
      D_loss.append(errD.item())

    # Write current model to checkpoint
    checkpoint_path = os.path.join(FLAGS.checkpoint, 
                                   "checkpoint" + str(epoch) + ".tar")
    torch.save({
      'epoch': epoch,
      'model_state_dictG': netG.state_dict(),
      'optimizer_state_dictG': optimizerG.state_dict(),
      'lossG': G_loss,
      'model_state_dictD': netD.state_dict(),
      'optimizer_state_dictD': optimizerD.state_dict(),
      'lossD': D_loss}, 
      checkpoint_path)

    # Check how the generator 
    util.write_images(epoch, FLAGS.results, fixed_noise, 16, 
        netG, device, use_fixed=FLAGS.use_fixed)

    logging.info("Time for epoch: " + str(time() - t1))


if __name__ == '__main__':
  main()
