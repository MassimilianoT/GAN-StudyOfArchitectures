import matplotlib.pyplot as plt
import os.path

dict = [
    {
        "NameOfGraph": "GAN - MNIST",
        "PathOfFile": "gan/models/mnist/generator_gan_mnist_log.txt",
        "OutputFile": "assets/log/MNIST-gan.png"
    },
    {
        "NameOfGraph": "GAN - CelebA",
        "PathOfFile": "gan/models/celeba/generator_gan_celeba_log.txt",
        "OutputFile": "assets/log/CelebA-gan.png"
    },
    {
        "NameOfGraph": "WGAN - MNIST",
        "PathOfFile": "wgan/models/mnist/generator_wgan_mnist_log.txt",
        "OutputFile": "assets/log/MNIST-wgan.png"
    },
    {
        "NameOfGraph": "WGAN - CelebA",
        "PathOfFile": "wgan/models/celeba/generator_wgan_celeba_log.txt",
        "OutputFile": "assets/log/CelebA-wgan.png"
    },
    {
        "NameOfGraph": "DCGAN - MNIST",
        "PathOfFile": "dcgan/models/mnist/generator_dcgan_mnist_log.txt",
        "OutputFile": "assets/log/MNIST-dcgan.png"
    },
    {
        "NameOfGraph": "DCGAN - CelebA",
        "PathOfFile": "dcgan/models/celeba/generator_dcgan_celeba_log.txt",
        "OutputFile": "assets/log/CelebA-dcgan.png"
    },
    {
        "NameOfGraph": "BEGAN - MNIST",
        "PathOfFile": "began/models/mnist/generator_began_mnist_log.txt",
        "OutputFile": "assets/log/MNIST-began.png"
    },
    {
        "NameOfGraph": "BEGAN - CelebA",
        "PathOfFile": "began/models/celeba/generator_began_celeba_log.txt",
        "OutputFile": "assets/log/CelebA-began.png"
    },
    {
        "NameOfGraph": "EBGAN - MNIST",
        "PathOfFile": "ebgan/models/mnist/generator_ebgan_mnist_log.txt",
        "OutputFile": "assets/log/MNIST-ebgan.png"
    },
    {
        "NameOfGraph": "EBGAN - CelebA",
        "PathOfFile": "ebgan/models/celeba/generator_ebgan_celeba_log.txt",
        "OutputFile": "assets/log/CelebA-ebgan.png"
    }
]

for log in dict:
    file = open(os.path.join(os.getcwd(), log['PathOfFile']))
    I = []
    G = []
    D = []
    for line in file.readlines():
        if "-" not in line:
            stripped = line.split(' ')
            I.append(float(stripped[0]))
            D.append(float(stripped[1]))
            G.append(float(stripped[2]))
    plt.figure()
    plt.plot(I, G, "-g", label="Generatore")
    plt.plot(I, D, "-r", label="Discriminatore")
    plt.legend(loc="upper left")
    plt.grid()
    plt.xlabel('Epoca')
    plt.ylabel('Loss Function')
    plt.title(log['NameOfGraph'])
    plt.savefig(os.path.join(os.getcwd(), log['OutputFile']))
