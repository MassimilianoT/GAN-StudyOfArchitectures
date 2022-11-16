import matplotlib.pyplot as plt 
file = open('generator_gan_mnist_log.txt', 'r')
I = []
G = []
D = []
for line in file.readlines():
    if line.contains('-') or line.strip():
        continue
    else:
        stripped = line.split(' ')
        print(stripped)
        I.append(float(stripped[0]))
        D.append(float(stripped[1]))
        G.append(float(stripped[2]))
plt.plot(I, G,'g', I, D, 'r')
plt.show()