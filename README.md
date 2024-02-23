# Unsupervised Artifact Detection in 3D OCTA Microvascular Imaging based on Generative Adversarial Networks
Official Pytorch implementation of OCTA-GAN, a novel Generative Adversarial Network (GAN) architecture for unsupervised optical coherence tomography angiography (OCTA) artifact detection as described in the paper.

### Framework:
OCTA-GAN's framework and model architecture is as follows:

<img width="512" alt="image" src="https://github.com/edsumpena/OCTA-GAN/assets/21966025/b83286e9-455d-4bc8-aac5-ffc94005ad12">
<br/><br/>
<img width="512" alt="image" src="https://github.com/edsumpena/OCTA-GAN/assets/21966025/0ebbd82a-38cc-4c75-a609-744302141c53">

### Dependencies

The following are the major dependencies for the model implementation and training:
```
python
torch
torchio (for 3D augmentations)
fvcore (for model complexity assessment)
typing
matplotlib
nvsmi (optional, diplays information about GPUs)
```

### Model Implementation:
- `PixelShuffle.py` - contains 3D implementation of subpixel convolution and SSMS Subpixel Convolution.
- `SingleShotMultiScale.py` - contains implementation of the SSMS layer for multi-scale processing.
- `OCTA_GAN_128.py` - contains implementation of the OCTA-GAN generator and discriminator architecture.
- `AlphaGAN_128.py` - contains implementation of the AlphaGAN generator and discriminator architecture used in the paper as 

### Training:
Update line 95 in `train.py` with the dataloader for your dataset:
```
image_datasets = {
    'train':    # TODO Add training dataloader
}
```
Then train the model by running the python script:
```
cd OCTA-GAN
python3 train.py
```
