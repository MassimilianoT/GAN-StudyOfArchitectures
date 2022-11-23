from gan.gan import *
from began.began import *
from dcgan.dcgan import *
from wgan.wgan import *
from ebgan.ebgan import *


def gan(dataset):
    train_GAN(True if dataset == 2 else False)


def began(dataset):
    train_BEGAN(True if dataset == 2 else False)


def dcgan(dataset):
    train_DCGAN(True if dataset == 2 else False)


def ebgan(dataset):
    train_EBGAN(True if dataset == 2 else False)


def wgan(dataset):
    train_wGAN(True if dataset == 2 else False)


def models_menu():
    print("-------------------------")
    print("Training of GAN Models")
    print("-------------------------")
    print("1. Train a GAN")
    print("2. Train a WGAN")
    print("3. Train a BEGAN")
    print("4. Train a DCGAN")
    print("5. Train a EBGAN")
    print()


def dataset_menu():
    print("-------------------------")
    print("Training on which dataset")
    print("-------------------------")
    print("1. Train on MNIST")
    print("2. Train on CelebA")
    print()


models_menu()
scelta = int(input("Scegli: "))
print("------------------------")

if scelta == 1:
    dataset_menu()
    scelta2 = int(input("Scegli: "))
    print("------------------------")
    if scelta2 == 1 or scelta2 == 2:
        gan(scelta2)
    else:
        print('errore')
elif scelta == 2:
    dataset_menu()
    scelta2 = int(input("Scegli: "))
    print("------------------------")
    if scelta2 == 1 or scelta2 == 2:
        wgan(scelta2)
    else:
        print('errore')
elif scelta == 3:
    dataset_menu()
    scelta2 = int(input("Scegli: "))
    print("------------------------")
    if scelta2 == 1 or scelta2 == 2:
        began(scelta2)
    else:
        print('errore')
elif scelta == 4:
    dataset_menu()
    scelta2 = int(input("Scegli: "))
    print("------------------------")
    if scelta2 == 1 or scelta2 == 2:
        dcgan(scelta2)
    else:
        print('errore')
elif scelta == 5:
    dataset_menu()
    scelta2 = int(input("Scegli: "))
    print("------------------------")
    if scelta2 == 1 or scelta2 == 2:
        ebgan(scelta2)
    else:
        print('errore')
else:
    print('error')
