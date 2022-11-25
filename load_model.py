from gan.gan import Generator as GAN_G
from gan.gan import img_shape as gan_shape
from began.began import Generator as BEGAN_G
from dcgan.dcgan import Generator as DCGAN_G
from wgan.wgan import Generator as WGAN_G
from wgan.wgan import img_shape as wgan_shape
from ebgan.ebgan import Generator as EBGAN_G
import torch
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image


def gan(dataset):
    latent_dim = 100
    if dataset == 2:
        generator = GAN_G(img_shape=(3, gan_shape[1], gan_shape[2]))
        generator.load_state_dict(torch.load('gan/models/celeba/generator_gan_celeba.pth'))
        generator.eval()
        path = 'assets/results/CelebA/generated_image_gan.png'
    else:
        generator = GAN_G()
        generator.load_state_dict(torch.load('gan/models/mnist/generator_gan_mnist.pth'))
        generator.eval()
        path = 'assets/results/Mnist/generated_image_gan.png'

    z = Variable(torch.Tensor(np.random.normal(0, 1, (25, latent_dim))))
    gen_image = generator(z)
    save_image(gen_image.data, path, normalize=True, nrow=5)
    print('Immagini create')


def began(dataset):
    latent_dim = 62
    if dataset == 2:
        generator = BEGAN_G(channels=3)
        generator.load_state_dict(torch.load('began/models/celeba/generator_began_celeba.pth'))
        generator.eval()
        path = 'assets/results/CelebA/generated_image_began.png'
    else:
        generator = BEGAN_G()
        generator.load_state_dict(torch.load('began/models/mnist/generator_began_mnist.pth'))
        generator.eval()
        path = 'assets/results/Mnist/generated_image_began.png'

    z = Variable(torch.FloatTensor(np.random.normal(0, 1, (25, latent_dim))))
    gen_image = generator(z)
    save_image(gen_image.data, path, normalize=True, nrow=5)
    print('Immagini create')


def dcgan(dataset):
    latent_dim = 100
    if dataset == 2:
        generator = DCGAN_G(channels=3)
        generator.load_state_dict(torch.load('dcgan/models/celeba/generator_dcgan_celeba.pth'))
        generator.eval()
        path = 'assets/results/CelebA/generated_image_dcgan.png'
    else:
        generator = DCGAN_G(channels=1)
        generator.load_state_dict(torch.load('dcgan/models/mnist/generator_dcgan_mnist.pth'))
        generator.eval()
        path = 'assets/results/Mnist/generated_image_dcgan.png'
    z = Variable(torch.FloatTensor(np.random.normal(0, 1, (25, latent_dim))))
    gen_image = generator(z)
    save_image(gen_image.data, path, normalize=True, nrow=5)
    print('Immagini create')


def ebgan(dataset):
    latent_dim = 62
    if dataset == 2:
        generator = EBGAN_G(channels=3)
        generator.load_state_dict(torch.load('ebgan/models/celeba/generator_ebgan_celeba.pth'))
        generator.eval()
        path = 'assets/results/CelebA/generated_image_ebgan.png'
    else:
        generator = EBGAN_G()
        generator.load_state_dict(torch.load('ebgan/models/mnist/generator_ebgan_mnist.pth'))
        generator.eval()
        path = 'assets/results/Mnist/generated_image_ebgan.png'
    z = Variable(torch.FloatTensor(np.random.normal(0, 1, (25, latent_dim))))
    gen_image = generator(z)
    save_image(gen_image.data, path, normalize=True, nrow=5)
    print('Immagini create')


def wgan(dataset):
    latent_dim = 100
    if dataset == 2:
        generator = WGAN_G(img_shape=(3, wgan_shape[1], wgan_shape[2]))
        generator.load_state_dict(torch.load('wgan/models/celeba/generator_wgan_celeba.pth'))
        generator.eval()
        path = 'assets/results/CelebA/generated_image_wgan.png'
    else:
        generator = WGAN_G(img_shape=wgan_shape)
        generator.load_state_dict(torch.load('wgan/models/mnist/generator_wgan_mnist.pth'))
        generator.eval()
        path = 'assets/results/Mnist/generated_image_wgan.png'
    z = Variable(torch.Tensor(np.random.normal(0, 1, (25, latent_dim))))
    gen_image = generator(z)
    save_image(gen_image.data, path, normalize=True, nrow=5)
    print('Immagini create')


def models_menu():
    print("-------------------------")
    print("Loading a GAN Models")
    print("-------------------------")
    print("1. Load a GAN")
    print("2. Load a WGAN")
    print("3. Load a BEGAN")
    print("4. Load a DCGAN")
    print("5. Load a EBGAN")
    print("6. Load ALL models")
    print()


def dataset_menu():
    print("-------------------------")
    print("Loading which dataset")
    print("-------------------------")
    print("1. Load MNIST")
    print("2. Load CelebA")
    print("3. Load ALL datasets")
    print()


models_menu()
scelta = int(input("Choose: "))
if scelta != 1 and scelta != 2 and scelta != 3 and scelta != 4 and scelta != 5 and scelta != 6:
    print('errore')
else:
    dataset_menu()
    scelta2 = int(input("Choose: "))
    print()
    if scelta2 != 1 and scelta2 != 2 and scelta2 != 3:
        print('errore')
    else:
        if scelta2 == 3:
            if scelta == 1:
                gan(1)
                gan(2)
            elif scelta == 2:
                wgan(1)
                wgan(2)
            elif scelta == 3:
                began(1)
                began(2)
            elif scelta == 4:
                dcgan(1)
                dcgan(2)
            elif scelta == 5:
                ebgan(1)
                ebgan(2)
            else:
                gan(1)
                gan(2)
                wgan(1)
                wgan(2)
                began(1)
                began(2)
                dcgan(1)
                dcgan(2)
                ebgan(1)
                ebgan(2)
        else:
            if scelta == 1:
                gan(scelta2)
            elif scelta == 2:
                wgan(scelta2)
            elif scelta == 3:
                began(scelta2)
            elif scelta == 4:
                dcgan(scelta2)
            elif scelta == 5:
                ebgan(scelta2)
            else:
                gan(scelta2)
                wgan(scelta2)
                began(scelta2)
                dcgan(scelta2)
                ebgan(scelta2)
