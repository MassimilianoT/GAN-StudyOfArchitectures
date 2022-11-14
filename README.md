# GAN-StudyOfArchitectures
Progetto per il corso "Machine Learning and Data Mining" per la laurea magistrale in Ingegneria Informatica all'Università degli studi di Brescia.
## Overview
In questo progetto abbiamo analizzato varie architetture per le reti GAN (Generative Adversarial Networks): le reti GAN appartengono alle reti generative, modelli di deep learning il cui scopo è produrre nuove istanze di dati che somiglino ad una distribuzione nota di dati. Le reti GAN risolvono questa task mediante un allenamento simultaneo di due modelli, il generatore G e il discriminatore D.
Le differenze nei vari modelli analizzati stanno nel modo in cui le reti sono allenate, nella loro architettura, negli ottimizzatori utilizzati e nella funzione di loss associata ai due modelli. 
## Observations
Di seguito si mostrano i risultati dei vari modelli allenati sui dataset Mnist e CelebA
### - Mnist
| ![GAN](./immagini%20generate%20dai%20%20modelli/Mnist/generated_image_gan.png) | ![WGAN](./immagini%20generate%20dai%20%20modelli/Mnist/generated_image_wgan.png) | ![BEGAN](./immagini%20generate%20dai%20%20modelli/Mnist/generated_image_began.png) | ![DCGAN](./immagini%20generate%20dai%20%20modelli/Mnist/generated_image_dcgan.png) | ![EBGAN](./immagini%20generate%20dai%20%20modelli/Mnist/generated_image_ebgan.png)
|:--:|:--:|:--:|:--:|:--:| 
| *GAN* | *WGAN* | *BEGAN* | *DCGAN* | *EBGAN* | 

### - CelebA
| ![GAN](./immagini%20generate%20dai%20%20modelli/CelebA/generated_image_gan.png) | ![WGAN](./immagini%20generate%20dai%20%20modelli/CelebA/generated_image_wgan.png) | ![BEGAN](./immagini%20generate%20dai%20%20modelli/CelebA/generated_image_began.png) | ![DCGAN](./immagini%20generate%20dai%20%20modelli/CelebA/generated_image_dcgan.png) | ![EBGAN](./immagini%20generate%20dai%20%20modelli/CelebA/generated_image_ebgan.png)
|:--:|:--:|:--:|:--:|:--:| 
| *GAN* | *WGAN* | *BEGAN* | *DCGAN* | *EBGAN* | 

