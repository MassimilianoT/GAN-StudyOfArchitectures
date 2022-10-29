from began import *

answer = input('Usare CelebA o Mnist? (C per celebA, M per Mnist)')
if answer is 'C':
    train_BEGAN(use_celebA=True)
elif answer is 'M':
    train_BEGAN()
else:
    print('Inserito carattere non corretto. Riprovare')