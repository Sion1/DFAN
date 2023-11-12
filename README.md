# DFAN


This repository contains the training and test code for the paper  "***Dual Feature Augmentation Network for
Generalized Zero-shot Learning***" accepted to BMVC 2023. 

![](figures/architecture.png)

## Running Environment
The implementation of **DFAN** is mainly based on Python 3.7.16 and [PyTorch](https://pytorch.org/) 1.12.1. To install all required dependencies:
```
$ pip install -r requirements.txt
```

### Training Script
```
$ python main.py --training --dataset DATA_SET --mat_path MAT_PATH
```


### Preparing Dataset and Model

We provide trained models ([Google Drive](https://drive.google.com/drive/folders/1PQkewCqlEl8FbgFOboB7WqmGgIvN95x9?usp=sharing)) on three different datasets: [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [SUN](http://cs.brown.edu/~gmpatter/sunattributes.html), [AWA2](http://cvml.ist.ac.at/AwA2/) in the CZSL/GZSL setting. 



## References
Parts of our codes based on:
* [FaisalAlamri0/ViT-ZSL](https://github.com/FaisalAlamri0/ViT-ZSL)

## Contact
If you have any questions about codes, please don't hesitate to contact us by xl294487391@gmail.com.
