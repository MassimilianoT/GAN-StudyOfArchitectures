from dcgan import *

answer = input('Usare CelebA o Mnist? (C per celebA, M per Mnist): ')
if answer is 'C':
    train_DCGAN(use_celebA=True)
elif answer is 'M':
    train_DCGAN(use_celebA=False)
else:
    print('Inserito carattere non corretto. Riprovare')
