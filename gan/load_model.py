from gan import Generator
import torch
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image

generator = Generator()
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()
z = Variable(torch.Tensor(np.random.normal(0, 1, (30, 100))))
gen_image = generator(z)
save_image(gen_image.data, 'generated_image.png')
print('Immagini create')
