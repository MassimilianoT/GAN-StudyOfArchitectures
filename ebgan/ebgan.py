import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

n_epochs = 1000
batch_size = 64
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_cpu = 8
latent_dim = 62
img_size = 32
#Vediamo come funziona con 64 invece che con 32

channels = 1
sample_interval = 400

img_shape = (channels, img_size, img_size)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, img_size = 32, latent_dim =62, channels =1):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, channels = 1, img_size = 32):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.channels = channels
        # Upsampling
        self.down = nn.Sequential(nn.Conv2d(self.channels, 64, 3, 2, 1), nn.ReLU())
        # Fully-connected layers
        self.down_size = self.img_size // 2
        down_dim = 64 * (self.img_size // 2) ** 2

        self.embedding = nn.Linear(down_dim, 32)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        # Upsampling
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, self.channels, 3, 1, 1))

    def forward(self, img):
        out = self.down(img)
        embedding = self.embedding(out.view(out.size(0), -1))
        out = self.fc(embedding)
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
        return out, embedding

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


def train_EBGAN(use_celebA = True, img_size = img_size):
    os.makedirs("images", exist_ok=True)
    #RGB
    if use_celebA:
        channels = 3
        os.makedirs("models/celeba", exist_ok=True)
        name_net = "models/celeba/generator_ebgan_celeba"
    else:
        channels = 1
        os.makedirs("models/mnist", exist_ok=True)
        name_net = "models/mnist/generator_ebgan_mnist"
    file_logger = open(name_net + '_log.txt', 'a')
    file_logger.write('Epoch - D Loss - G Loss\n')

    # Reconstruction loss of AE
    pixelwise_loss = nn.MSELoss()

    # Initialize generator and discriminator
    generator = Generator(img_size=img_size, channels=channels)
    discriminator = Discriminator(img_size=img_size, channels=channels)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        pixelwise_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Configure data loader
    dataloader = get_dataloader(use_celebA=use_celebA, img_size=img_size)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    def pullaway_loss(embeddings):
        norm = torch.sqrt(torch.sum(embeddings ** 2, -1, keepdim=True))
        normalized_emb = embeddings / norm
        similarity = torch.matmul(normalized_emb, normalized_emb.transpose(1, 0))
        batch_size = embeddings.size(0)
        loss_pt = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
        return loss_pt


    # ----------
    #  Training
    # ----------

    # BEGAN hyper parameters
    lambda_pt = 0.1
    margin = max(1, batch_size / 64.0)

    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)
            recon_imgs, img_embeddings = discriminator(gen_imgs)

            # Loss measures generator's ability to fool the discriminator
            g_loss = pixelwise_loss(recon_imgs, gen_imgs.detach()) + lambda_pt * pullaway_loss(img_embeddings)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_recon, _ = discriminator(real_imgs)
            fake_recon, _ = discriminator(gen_imgs.detach())

            d_loss_real = pixelwise_loss(real_recon, real_imgs)
            d_loss_fake = pixelwise_loss(fake_recon, gen_imgs.detach())

            d_loss = d_loss_real
            if (margin - d_loss_fake.data).item() > 0:
                d_loss += margin - d_loss_fake

            d_loss.backward()
            optimizer_D.step()

            # --------------
            # Log Progress
            # --------------

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            file_logger.write('%d %f %f\n' % (epoch, d_loss.item(), g_loss.item()))
            file_logger.flush()
            if epoch % 100 == 0:
                torch.save(generator.state_dict(), name_net + '_%d.pth' % epoch)
    torch.save(generator.state_dict(), name_net + '.pth')
    file_logger.close()