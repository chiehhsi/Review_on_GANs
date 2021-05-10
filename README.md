# Review on GAN Models with Unpaired Dataset

Generative Adversarial Networks (GAN) are the model that widely used to solve the image-to-image translation mapping an image from a source domain to a target domain. The accessible fields including colorization, super-resolution, style transfer, etc. We did the research review focusing on those who implemented GANs on unpaired datasets. Although there are plenty of GANs, we chose **CycleGAN**, **AttentionGAN**, and **U-GAN-IT** among all.

This repository is mainly for learning and record use. Despite the details in existing repositories, for easy use, we wrote the simple instructions how we implemented them, translated images we obtained from these models and also provided related resources for GANs. Here we worked on `Horse2zebra` dataset.

For original code and implementation, please visit original repositories : [Pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [AttentionGAN](https://github.com/Ha0Tang/AttentionGAN), [U-GAN-IT](https://github.com/znxlwm/UGATIT-pytorch).

### Review Models

1. CycleGAN 
- Package Setup
```
$ git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
$ cd pytorch-CycleGAN-and-pix2pix
```
- Prepare Dataset
- Load Trained Model
- Test Model & Results



2. AttentionGAN
- Package Setup
- Prepare Dataset
- Load Trained Model
- Test Model & Results

3. U-GAN-IT
- Package Setup
- Prepare Dataset
- Load Trained Model
- Test Model & Results
We trained the model from scratch due to the limitation of the hardware we have, instead of training the model 1,000,000 epochs like default, the pretrained model we had was only trained under 12,000 epochs.


### Our Implemented Results (Based on Custom Horse Dataset)



### Repository Files
- `Review_Report.pdf` : Details the conclusion that we have after implementing and comparing these three GAN models
- Pretained Model : Need to put

### Related Models
- BycleGAN
- AsymmetricGAN
- 


### Reference
- J-Y. Zhu, T. Park, P. Isola, and A. A Efros, "*Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*", Computer Vision (ICCV), 2017 IEEE International Conference
-  H. Tang, H. Liu, D. Xu, P. H. S. Torr and N. Sebe, “*AttentionGAN: Unpaired Image-to-Image Translation using Attention-Guided Generative Adversarial Networks*”, in CoRR, abs/1911.11897, 2019
- J. Kim, M. Kim, H. Kang and K. Lee, “*U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation*” arXiv 1907.10830, 2020


### Constributors
[Chieh-Hsi Lin](https://github.com/chiehhsi), [HyeongHwan Kwon](https://github.com/hkwon31)


### To Do List

- [ ] Add README.md
- [ ] Add command instruction for each github repositories
- [ ] Upload Trained Model
- [ ] Add Predicted Results

-AttentionGAN : https://github.com/Ha0Tang/AttentionGAN
-pytorch CycleGAN : https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
-U-GAN-IT : https://github.com/znxlwm/UGATIT-pytorch
