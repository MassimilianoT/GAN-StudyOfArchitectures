import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

n_epochs= 200
batch_size = 64
lr = 0.00005
n_cpu = 8 
latent_dim = 100
img_size = 28
channels = 1
n_critic = 5
clip_value = 0.01
sample_interval = 400

img_shape = (channels, img_size, img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1,28,28)):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self,img_shape=(1, 28, 28)):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

# Metodo per determinare se usare CelebA o Mnist
def get_dataloader(use_celebA=True, img_size=img_size):
    if use_celebA:
        os.makedirs("../data/celeba", exist_ok=True)
        dataloader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                "../data/celeba",
                transform=transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ])
            ),
            batch_size=batch_size,
            shuffle=True,
        )
    else:
        os.makedirs("../data/mnist", exist_ok=True)
        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "../data/mnist",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
    return dataloader

def train_wGAN(use_celebA = True):
    os.makedirs("images", exist_ok=True)
    global img_shape
    if use_celebA:
        img_shape = (3, img_size, img_size)
        

    # Initialize generator and discriminator
    generator = Generator(img_shape=img_shape)
    discriminator = Discriminator(img_shape=img_shape)

    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Configure data loader
    dataloader = get_dataloader(use_celebA=use_celebA, img_size=img_size)

    # Optimizers
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    batches_done = 0
    for epoch in range(n_epochs):

        for i, (imgs, _) in enumerate(dataloader):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

            # Generate a batch of images
            fake_imgs = generator(z).detach()
            # Adversarial loss
            loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

            # Train the generator every n_critic iterations
            if i % n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = generator(z)
                # Adversarial loss
                loss_G = -torch.mean(discriminator(gen_imgs))

                loss_G.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
                )

            if batches_done % sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            batches_done += 1
    if use_celebA:
        name_net = "models/generator_wgan_celeba.pth"
    else:
        name_net = "models/generator_wgan_mnist.pth"
    torch.save(generator.state_dict(), name_net)   
