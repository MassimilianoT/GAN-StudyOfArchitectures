from ebgan import *

answer = input('Usare CelebA o Mnist? (C per celebA, M per Mnist)')
if answer is 'C':
    train_EBGAN(use_celebA=True)
elif answer is 'M':
    train_EBGAN()
else:
    print('Inserito carattere non corretto. Riprovare')