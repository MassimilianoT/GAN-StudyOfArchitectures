from dcgan import Generator
import torch
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image

latent_dim = 100
usecelebA = True
usecelebA = input('Inserire M per Mnist e C per CelebA (C default)') is 'C' 
if usecelebA:
    generator = Generator(channels=3)
    generator.load_state_dict(torch.load('models/celeba/generator_dcgan_celeba.pth'))
    generator.eval()
else:
    generator = Generator(channels=1)
    generator.load_state_dict(torch.load('models/mnist/generator_dcgan_mnist.pth'))
    generator.eval()

z = Variable(torch.FloatTensor(np.random.normal(0,1, (25, latent_dim))))
gen_image = generator(z)
save_image(gen_image.data, 'generated_image.png', normalize=True, nrow=5)
print('Immagini create')
