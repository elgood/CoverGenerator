import argparse
import torch
from torchvision import transforms
from torchvision import datasets

def get_dataloader(batch_size, image_size, data_dir):
  """ Creates a data loader to read in the images.
  """

  transform = transforms.Compose([transforms.Resize(image_size),
                                  transforms.ToTensor()]
  train_dataset = datasets.ImageFolder(data_dir, transform)
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=batch_size, shuffle=True)

  return train_loader



