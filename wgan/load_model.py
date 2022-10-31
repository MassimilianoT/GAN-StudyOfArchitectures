from locale import normalize
from wgan import *
import torch
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image

latent_dim=100
usecelebA = True
usecelebA = input('Inserire M per Mnist e C per CelebA (C default)') is 'C' 
if usecelebA:
    global img_shape
    img_shape = (3,img_shape[1], img_shape[2])
    generator = Generator(img_shape=img_shape)
    generator.load_state_dict(torch.load('models/generator_wgan_celeba.pth'))
    generator.eval()
else:
    generator = Generator(img_shape=img_shape)
    generator.load_state_dict(torch.load('models/generator_wgan_mnist.pth'))
    generator.eval()
    
z = Variable(torch.Tensor(np.random.normal(0, 1, (30, latent_dim))))
gen_image = generator(z)
save_image(gen_image.data, 'generated_image.png', normalize=True, nrow=5)
print('Immagini create')