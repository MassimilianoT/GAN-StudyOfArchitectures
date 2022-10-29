from wgan import *

answer = input('Usare CelebA o Mnist? (C per celebA, M per Mnist)')
if answer is 'C':
    train_wGAN(use_celebA=True)
elif answer is 'M':
    train_wGAN()
else:
    print('Inserito carattere non corretto. Riprovare')
