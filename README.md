# GAN-StudyOfArchitectures

Progetto per il corso "Machine Learning and Data Mining" per la laurea magistrale in Ingegneria Informatica all'Università degli studi di Brescia.

## Indice

 - [Introduzione](#introduzione)
	 - [Modelli Generativi](#modelli-generativi)
	 - [Rete Discriminante](#rete-discriminante)
 - [Architetture](#architetture)
	 - [BEGAN](#began)
	 - [DCGAN](#dcgan)
	 - [EBGAN](#ebgan)
	 - [GAN](#gan)
	 - [WGAN](#wgan)
 - [Risultati](#risultati)
	 - [MNIST](#mnist)
	 - [CelebA](#celeba)

## Introduzione

Il progetto in questione aveva l'obiettivo di mettere a confronto diverse architetture di reti GAN per valutare le loro performance su due diversi dataset (**MNIST** e **CelebA**). Le architetture che abbiamo testato sono:

 - **BEGAN** (_Boundary Equilibrium Generative Adversarial Networks_)
 - **DCGAN** (_Deep Convolutional Generative Adversarial Network_)
 - **EBGAN** (_Energy-based Generative Adversarial Network_)
 - **GAN** (_Generative Adversarial Networks_)
 - **WGAN** (_Wasserstein Generative Adversarial Networks_)

Nei capitoli di questo documento andremo a descrivere ciò che si intende per rete generativa e come differiscono le GAN da questa definizione iniziale, specificandosi poi nelle diverse architetture (ognuna con le proprie differenze rispetto alla rete GAN base).

### Modelli Generativi

L'obiettivo dei modelli generativi è imparare un modello che rappresenta la distribuzione dei dati di training che gli vengono dati in input

## Architetture

### BEGAN

_BEGAN: Boundary Equilibrium Generative Adversarial Networks_

#### Autori

David Berthelot, Thomas Schumm, Luke Metz

#### Descrizione

### DCGAN

_Deep Convolutional Generative Adversarial Network_

#### Autori

Alec Radford, Luke Metz, Soumith Chintala

#### Descrizione

### EBGAN

_Energy-based Generative Adversarial Network_

#### Autori

Junbo Zhao, Michael Mathieu, Yann LeCun

#### Descrizione

### GAN

_Generative Adversarial Network_

#### Autori

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio

#### Descrizione

### WGAN

_Wasserstein Generative Adversarial Network_

#### Autori

Martin Arjovsky, Soumith Chintala, Léon Bottou

#### Descrizione

## Risultati

Di seguito si mostrano i risultati dei vari modelli allenati sui dataset MNIST e CelebA

### MNIST
|![CIAO](https://upload.wikimedia.org/wikipedia/commons/6/6f/Rete_generativa_avversaria.png) | ![Ciao](https://upload.wikimedia.org/wikipedia/commons/6/6f/Rete_generativa_avversaria.png) |![Ciao](https://upload.wikimedia.org/wikipedia/commons/6/6f/Rete_generativa_avversaria.png)|![Ciao](https://upload.wikimedia.org/wikipedia/commons/6/6f/Rete_generativa_avversaria.png)|![Ciao](https://upload.wikimedia.org/wikipedia/commons/6/6f/Rete_generativa_avversaria.png)|
|--|--|--|--|--|
|***GAN***|***WGAN***|***BEGAN***|***DCGAN***|***EBGAN***|

### CelebA
|![CIAO](https://upload.wikimedia.org/wikipedia/commons/6/6f/Rete_generativa_avversaria.png) | ![Ciao](https://upload.wikimedia.org/wikipedia/commons/6/6f/Rete_generativa_avversaria.png) |![Ciao](https://upload.wikimedia.org/wikipedia/commons/6/6f/Rete_generativa_avversaria.png)|![Ciao](https://upload.wikimedia.org/wikipedia/commons/6/6f/Rete_generativa_avversaria.png)|![Ciao](https://upload.wikimedia.org/wikipedia/commons/6/6f/Rete_generativa_avversaria.png)|
|--|--|--|--|--|
|***GAN***|***WGAN***|***BEGAN***|***DCGAN***|***EBGAN***|


&copy; Glisenti Mirko, Tummolo Massimiliano - 2022