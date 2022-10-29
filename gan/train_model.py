from gan import *
answer = input('Usare CelebA o Mnist? (C per celebA, M per Mnist)')
if answer is 'C':
    train_GAN(use_celebA=True)
elif answer is 'M':
    train_GAN()
else:
    print('Inserito carattere non corretto. Riprovare')
