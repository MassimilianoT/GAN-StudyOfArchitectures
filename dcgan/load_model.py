from dcgan import Generator
import torch
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image

latent_dim = 100
generator = Generator()
generator.load_state_dict(torch.load('generator_dcgan.pth'))
generator.eval()
z = Variable(torch.FloatTensor(np.random.normal(0,2, (100, latent_dim))))
gen_image = generator(z)
save_image(gen_image.data[:25], 'generated_image.png', normalize=True, nrow=5)
print('Immagini create')
