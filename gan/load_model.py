from gan import Generator
import torch
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image

latent_dim=100
usecelebA = True
usecelebA = input('Inserire M per Mnist e C per CelebA (C default)') is 'C' 
if usecelebA:
    generator = Generator(channels=3)
    generator.load_state_dict(torch.load('/models/generator_gan_celeba.pth'))
    generator.eval()
else:
    generator = Generator()
    generator.load_state_dict(torch.load('/models/generator_gan_mnist.pth'))
    generator.eval()
    
z = Variable(torch.Tensor(np.random.normal(0, 1, (30, latent_dim))))
gen_image = generator(z)
save_image(gen_image.data, 'generated_image.png')
print('Immagini create')
