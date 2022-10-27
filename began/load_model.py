from began import Generator
import torch
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image

latent_dim = 62
generator = Generator()
generator.load_state_dict(torch.load('generator_began.pth'))
generator.eval()
z = Variable(torch.FloatTensor(np.random.normal(0,2, (25, latent_dim))))
gen_image = generator(z)
save_image(gen_image.data, 'generated_image.png')
print('Immagini create')
