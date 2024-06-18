import random
import numpy as np
import torch.nn as nn
import torchio as tio
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch

from PIL import Image

from OCTA_GAN_128 import *

deterministic = True

if deterministic:
    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

#
# GPU CPU - nice way to setup the device as it works on any machine
#
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'Device is {device}')

if device == 'cuda':
    print(f'CUDA device {torch.cuda.device(0)}')
    print(f'Number of devices: {torch.cuda.device_count()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')


# Read in data locations and labels from FHS dataset
print('Loading dataset...')

data_transforms = {
    
    'train': tio.Compose([
        tio.RescaleIntensity(percentiles=(0.5, 99.5), out_min_max=(-1, 1)),
    ]),
    'artifact': tio.Compose([
        tio.RescaleIntensity(percentiles=(0.5, 99.5), out_min_max=(-1, 1)),
    ]),
}


image_datasets = {
    'train':    # TODO Add dataloader to good-quality volumes
    'artifact_0':   # TODO Add dataloader to poor-quality volumes
    'artifact_1':   # TODO Add dataloader to questionable-quality volumes
}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=(not deterministic), num_workers=2) for x in ['train', 'artifact_0', 'artifact_1']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'artifact_0', 'artifact_1']}

def inf_train_gen(data_loader):
    while True:
        for _, images in enumerate(data_loader):
            yield images

# Specify model layer(s) to apply Grad-CAM
cam_conv_layer = [2]

data_points = len(cam_conv_layer)
use_gen = True
latent_dim = 1024

# Load OCTA-GAN model(s)
D = Discriminator()
D = nn.DataParallel(D)
D.to(device)

if use_gen:
    G = Generator()
    G = nn.DataParallel(G)
    G.to(device)


D.load_state_dict(torch.load('./path-to-discriminator', map_location=torch.device('cpu')))

if use_gen:
    G.load_state_dict(torch.load('./path-to-generator', map_location=torch.device('cpu')))

print(D)


gradient = None
activations = None

# Grad-CAM implementation for non-classification tasks
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradient = []
        self.activations = []

        self.score = None

        self.model.eval()

        self.hook_gradients()
        self.hook_activations()
    
    def hook_gradients(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradient.append(grad_output[0])
            print("conv: " + str(grad_output[0].size()))

        for _, module in self.model.module.named_modules():
            if isinstance(module, nn.Conv3d):
                module.register_full_backward_hook(backward_hook)

    def hook_activations(self):
        def forward_hook(module, input, output):
            self.activations.append(output.detach())
            print("activation: " + str(output.detach().size()))

        for _, module in self.model.module.named_modules():
            if isinstance(module, nn.LeakyReLU) or isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook)

    # Generate layer-wise Grad-CAM
    def generate_gradcam(self, input_tensor, layer):
        self.gradient = []
        self.activations = []
        
        self.model.zero_grad()

        # Take the output discriminator score and backpropogate to 
        # obtain gradients for each layer
        output = self.model(input_tensor)
        one_hot = torch.zeros_like(output)
        one_hot[0,0,0,0,0] = output.mean().data

        self.score = output.mean()
        score = output.mean()

        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradient[-layer].mean(dim=(2, 3, 4), keepdim=True)
        activations = self.activations[layer - 1]

        print(gradients.size())
        print(activations.size())

        cam = (gradients * activations).sum(dim=1, keepdim=True)
        cam = torch.abs(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize between 0 and 1

        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='trilinear')

        cam = cam.cpu().numpy()

        return cam, score
    
    # FullGradCAM - compute gradients with respect to input volume to highest
    # resolution class activation map
    def generate_fullgradcam(self, input_tensor):
        self.model.zero_grad()

        input_tensor.requires_grad = True

        # Forward pass through the discriminator
        discriminator_output = self.model(input_tensor)

        self.score = discriminator_output.mean()
        score = discriminator_output.mean()

        # Compute gradients of the discriminator output with respect to the input volume
        gradients = torch.autograd.grad(outputs=discriminator_output, inputs=input_tensor,
                                        grad_outputs=torch.ones_like(discriminator_output),
                                        create_graph=True, retain_graph=True)[0]

        # Normalize gradients across spatial dimensions
        normalized_gradients = torch.mean(torch.abs(gradients), dim=(2, 3, 4), keepdim=True)

        print(normalized_gradients.size())

        # Multiply gradients with input volume to obtain importance maps
        importance_maps = normalized_gradients * input_tensor

        print(importance_maps.size())

        # Upsample importance maps to match input volume size
        upsampled_maps = F.interpolate(importance_maps, size=input_tensor.shape[2:], mode='trilinear')

        # Combine importance maps from all layers
        fullgrad_map = torch.sum(upsampled_maps, dim=1, keepdim=True)

        input_tensor.requires_grad = False

        fullgrad_map = fullgrad_map.detach().cpu().numpy()
        fullgrad_map = (fullgrad_map - fullgrad_map.min()) / (fullgrad_map.max() - fullgrad_map.min())

        return fullgrad_map, score

    
    def get_score(self):
        return self.score

grad_cam = GradCAM(D)


gen_load_normal = inf_train_gen(image_datasets['train'])
gen_load_artifact0 = inf_train_gen(image_datasets['artifact_0'])
gen_load_artifact1 = inf_train_gen(image_datasets['artifact_1'])

if use_gen:
    rand = Variable(torch.randn((1, latent_dim))).to(device)
    img_rand = G(rand).detach()

img_normal = gen_load_normal.__next__()
img_artifact0 = gen_load_artifact0.__next__()
img_artifact1 = gen_load_artifact1.__next__()

img_norm_label = img_normal[1:3]
img_artifact0_label = img_artifact0[1:3]
img_artifact1_label = img_artifact1[1:3]

img_normal = img_normal[0]
img_artifact0 = img_artifact0[0]
img_artifact1 = img_artifact1[0]

img_normal = Variable(img_normal[None,],volatile=True).to(device, non_blocking=True)
img_artifact0 = Variable(img_artifact0[None,],volatile=True).to(device, non_blocking=True)
img_artifact1 = Variable(img_artifact1[None,],volatile=True).to(device, non_blocking=True)

for i in range(data_points):

    input_image = img_artifact0
    
    cam = grad_cam.generate_gradcam(input_image, cam_conv_layer[i])

    scr = cam[1]
    cam = cam[0]

    min = cam.min()
    max = cam.max()

    # Generate Grad-CAMs and FullGradCAMs for poor-quality volumes

    print('\n\nQuality 0 Volume:')
    print(scr)
    print(img_artifact0_label)

    plt.figure(figsize=(20, 6))
    for ii in range(10):
        plt.subplot(2,5,ii + 1)

        img = Image.fromarray(cam[0,0,128 // 10 * ii,:,:])

        plt.imshow(input_image[0,0,128 // 10 * ii,:,:], cmap='gray', clim=(-1, 1))
        plt.imshow(np.array(img), cmap = 'jet', alpha=0.5, clim=(min, max))
        plt.colorbar()
    plt.savefig('./Gen_test_outputs/Grad CAM/MIP/Grad-CAM_quality0_' + str(i) + '_norm.png')
    plt.close()

    plt.figure(figsize=(20, 6))
    for ii in range(10):
        plt.subplot(2,5,ii + 1)

        img = Image.fromarray(cam[0,0,128 // 10 * ii,:,:])

        plt.imshow(input_image[0,0,128 // 10 * ii,:,:], cmap='gray', clim=(-1, 1))
        plt.imshow(np.array(img), cmap = 'jet', alpha=0.5)
        plt.colorbar()
    plt.savefig('./Gen_test_outputs/Grad CAM/MIP/Grad-CAM_quality0_' + str(i) + '.png')
    plt.close()

    img = (input_image.cpu().numpy() + 1) / 2
    mip_img = np.true_divide(img.sum(3),(img >= 0.01).sum(3))
    mip_cam = np.true_divide(cam.sum(3),(cam >= 0.01).sum(3))

    plt.imshow(mip_img[0,0,:,:], cmap = 'gray')
    plt.imshow(mip_cam[0,0,:,:], cmap = 'jet', alpha=0.5)
    plt.colorbar()
    plt.savefig('./Gen_test_outputs/Grad CAM/MIP/MIP_Grad-CAM_quality0_' + str(i) + '.png')
    plt.close()


    fullcam = grad_cam.generate_fullgradcam(input_image)

    scr = fullcam[1]
    fullcam = fullcam[0]

    min = fullcam.min()
    max = fullcam.max()

    print('\nFullGradCAM:')
    print(scr)

    plt.figure(figsize=(22, 6))
    for ii in range(10):
        plt.subplot(2,5,ii + 1)

        img = fullcam[0,0,128 // 10 // (128 // fullcam.shape[2]) * ii,:,:]

        plt.imshow(input_image[0,0,128 // 10 * ii,:,:], cmap='gray', clim=(-1, 1))
        plt.imshow(img, cmap = 'jet', alpha=0.5)
        plt.colorbar()
    plt.savefig('./Gen_test_outputs/Grad CAM/MIP/Full-Grad-CAM_quality0_' + str(i) + '.png')
    plt.close()

    img = (input_image.cpu().numpy() + 1) / 2
    mip_img = np.true_divide(img.sum(3),(img >= 0.01).sum(3))
    mip_cam = np.true_divide(fullcam.sum(3),(fullcam >= 0.01).sum(3))

    plt.imshow(mip_img[0,0,:,:], cmap = 'gray')
    plt.imshow(mip_cam[0,0,:,:], cmap = 'jet', alpha=0.5)
    plt.colorbar()
    plt.savefig('./Gen_test_outputs/Grad CAM/MIP/MIP_Full-Grad-CAM_quality0_' + str(i) + '.png')
    plt.close()

    plt.figure(figsize=(22, 6))
    for ii in range(10):
        plt.subplot(2,5,ii + 1)
        plt.imshow(input_image[0,0,128 // 10 * ii,:,:], cmap='gray', clim=(-1, 1))
        plt.colorbar()
    plt.savefig('./Gen_test_outputs/Grad CAM/MIP/Original_quality0_' + str(i) + '.png')
    plt.close()

    plt.imshow(mip_img[0,0,:,:], cmap = 'gray')
    plt.colorbar()
    plt.savefig('./Gen_test_outputs/Grad CAM/MIP/Original-mip_quality0_' + str(i) + '.png')
    plt.close()





    # Generate Grad-CAMs and FullGradCAMs for questionable-quality volumes

    input_image = img_artifact1
    
    cam = grad_cam.generate_gradcam(input_image, cam_conv_layer[i])

    scr = cam[1]
    cam = cam[0]

    min = cam.min()
    max = cam.max()

    print('\n\nQuality 1 Volume:')
    print(scr)
    print(img_artifact1_label)

    plt.figure(figsize=(20, 6))
    for ii in range(10):
        plt.subplot(2,5,ii + 1)

        img = Image.fromarray(cam[0,0,128 // 10 * ii,:,:])

        plt.imshow(input_image[0,0,128 // 10 * ii,:,:], cmap='gray', clim=(-1, 1))
        plt.imshow(np.array(img), cmap = 'jet', alpha=0.5, clim=(min, max))
        plt.colorbar()
    plt.savefig('./Gen_test_outputs/Grad CAM/MIP/Grad-CAM_quality1_' + str(i) + '_norm.png')
    plt.close()

    plt.figure(figsize=(20, 6))
    for ii in range(10):
        plt.subplot(2,5,ii + 1)

        img = Image.fromarray(cam[0,0,128 // 10 * ii,:,:])

        plt.imshow(input_image[0,0,128 // 10 * ii,:,:], cmap='gray', clim=(-1, 1))
        plt.imshow(np.array(img), cmap = 'jet', alpha=0.5)
        plt.colorbar()
    plt.savefig('./Gen_test_outputs/Grad CAM/MIP/Grad-CAM_quality1_' + str(i) + '.png')
    plt.close()

    img = (input_image.cpu().numpy() + 1) / 2
    mip_img = np.true_divide(img.sum(3),(img >= 0.01).sum(3))
    mip_cam = np.true_divide(cam.sum(3),(cam >= 0.01).sum(3))

    plt.imshow(mip_img[0,0,:,:], cmap = 'gray')
    plt.imshow(mip_cam[0,0,:,:], cmap = 'jet', alpha=0.5)
    plt.colorbar()
    plt.savefig('./Gen_test_outputs/Grad CAM/MIP/MIP_Grad-CAM_quality1_' + str(i) + '.png')
    plt.close()


    fullcam = grad_cam.generate_fullgradcam(input_image)

    scr = fullcam[1]
    fullcam = fullcam[0]

    min = fullcam.min()
    max = fullcam.max()

    print('\nFullGradCAM:')
    print(scr)

    plt.figure(figsize=(22, 6))
    for ii in range(10):
        plt.subplot(2,5,ii + 1)

        img = fullcam[0,0,128 // 10 // (128 // fullcam.shape[2]) * ii,:,:]

        plt.imshow(input_image[0,0,128 // 10 * ii,:,:], cmap='gray', clim=(-1, 1))
        plt.imshow(img, cmap = 'jet', alpha=0.5)
        plt.colorbar()
    plt.savefig('./Gen_test_outputs/Grad CAM/MIP/Full-Grad-CAM_quality1_' + str(i) + '.png')
    plt.close()

    img = (input_image.cpu().numpy() + 1) / 2
    mip_img = np.true_divide(img.sum(3),(img >= 0.01).sum(3))
    mip_cam = np.true_divide(fullcam.sum(3),(fullcam >= 0.01).sum(3))

    plt.imshow(mip_img[0,0,:,:], cmap = 'gray')
    plt.imshow(mip_cam[0,0,:,:], cmap = 'jet', alpha=0.5)
    plt.colorbar()
    plt.savefig('./Gen_test_outputs/Grad CAM/MIP/MIP_Full-Grad-CAM_quality1_' + str(i) + '.png')
    plt.close()

    plt.figure(figsize=(22, 6))
    for ii in range(10):
        plt.subplot(2,5,ii + 1)
        plt.imshow(input_image[0,0,128 // 10 * ii,:,:], cmap='gray', clim=(-1, 1))
        plt.colorbar()
    plt.savefig('./Gen_test_outputs/Grad CAM/MIP/Original_quality1_' + str(i) + '.png')
    plt.close()

    plt.imshow(mip_img[0,0,:,:], cmap = 'gray')
    plt.colorbar()
    plt.savefig('./Gen_test_outputs/Grad CAM/MIP/Original-mip_quality1_' + str(i) + '.png')
    plt.close()




    # Generate Grad-CAMs and FullGradCAMs for good-quality volumes

    input_image = img_normal
    
    cam = grad_cam.generate_gradcam(input_image, cam_conv_layer[i])

    scr = cam[1]
    cam = cam[0]

    min = cam.min()
    max = cam.max()

    print('\n\nQuality 2 Volume:')

    plt.figure(figsize=(20, 6))
    for ii in range(10):
        plt.subplot(2,5,ii + 1)

        img = Image.fromarray(cam[0,0,128 // 10 * ii,:,:])

        plt.imshow(input_image[0,0,128 // 10 * ii,:,:], cmap='gray', clim=(-1, 1))
        plt.imshow(np.array(img), cmap = 'jet', alpha=0.5)
        plt.colorbar()
    plt.savefig('./Gen_test_outputs/Grad CAM/MIP/Grad-CAM_quality2_' + str(i) + '_norm.png')
    plt.close()

    plt.figure(figsize=(20, 6))
    for ii in range(10):
        plt.subplot(2,5,ii + 1)

        img = Image.fromarray(cam[0,0,128 // 10 * ii,:,:])

        plt.imshow(input_image[0,0,128 // 10 * ii,:,:], cmap='gray', clim=(-1, 1))
        # plt.imshow(np.array(img), cmap = 'jet', alpha=0.5, clim=(min, max))
        plt.imshow(np.array(img), cmap = 'jet', alpha=0.5)
        plt.colorbar()
    plt.savefig('./Gen_test_outputs/Grad CAM/MIP/Grad-CAM_quality2_' + str(i) + '.png')
    plt.close()

    img = (input_image.cpu().numpy() + 1) / 2
    mip_img = np.true_divide(img.sum(3),(img >= 0.01).sum(3))
    mip_cam = np.true_divide(cam.sum(3),(cam >= 0.01).sum(3))

    plt.imshow(mip_img[0,0,:,:], cmap = 'gray')
    plt.imshow(mip_cam[0,0,:,:], cmap = 'jet', alpha=0.5)
    plt.colorbar()
    plt.savefig('./Gen_test_outputs/Grad CAM/MIP/MIP_Grad-CAM_quality2_' + str(i) + '.png')
    plt.close()



    fullcam = grad_cam.generate_fullgradcam(input_image)

    scr = fullcam[1]
    fullcam = fullcam[0]

    min = fullcam.min()
    max = fullcam.max()

    print('\nFullGradCAM:')
    print(scr)

    plt.figure(figsize=(22, 6))
    for ii in range(10):
        plt.subplot(2,5,ii + 1)

        img = fullcam[0,0,128 // 10 // (128 // fullcam.shape[2]) * ii,:,:]

        plt.imshow(input_image[0,0,128 // 10 * ii,:,:], cmap='gray', clim=(-1, 1))
        plt.imshow(img, cmap = 'jet', alpha=0.5)
        plt.colorbar()
    plt.savefig('./Gen_test_outputs/Grad CAM/MIP/Full-Grad-CAM_quality2_' + str(i) + '.png')
    plt.close()

    img = (input_image.cpu().numpy() + 1) / 2
    mip_img = np.true_divide(img.sum(3),(img >= 0.01).sum(3))
    mip_cam = np.true_divide(fullcam.sum(3),(fullcam >= 0.01).sum(3))

    plt.imshow(mip_img[0,0,:,:], cmap = 'gray')
    plt.imshow(mip_cam[0,0,:,:], cmap = 'jet', alpha=0.5)
    plt.colorbar()
    plt.savefig('./Gen_test_outputs/Grad CAM/MIP/MIP_Full-Grad-CAM_quality2_' + str(i) + '.png')
    plt.close()


    plt.figure(figsize=(22, 6))
    for ii in range(10):
        plt.subplot(2,5,ii + 1)
        plt.imshow(input_image[0,0,128 // 10 * ii,:,:], cmap='gray', clim=(-1, 1))
        plt.colorbar()
    plt.savefig('./Gen_test_outputs/Grad CAM/MIP/Original_quality2_' + str(i) + '.png')
    plt.close()

    plt.imshow(mip_img[0,0,:,:], cmap = 'gray')
    plt.colorbar()
    plt.savefig('./Gen_test_outputs/Grad CAM/MIP/Original-mip_quality2_' + str(i) + '.png')
    plt.close()


    # Generate Grad-CAMs and FullGradCAMs for synthetic OCTA-GAN volumes

    if use_gen:
        input_image = img_rand
        
        cam = grad_cam.generate_gradcam(input_image, cam_conv_layer[i])

        scr = cam[1]
        cam = cam[0]

        min = cam.min()
        max = cam.max()

        print('\n\nGenerator Volume:')
        print(scr)

        plt.figure(figsize=(20, 6))
        for ii in range(10):
            plt.subplot(2,5,ii + 1)

            img = Image.fromarray(cam[0,0,128 // 10 * ii,:,:])

            plt.imshow(input_image[0,0,128 // 10 * ii,:,:], cmap='gray', clim=(-1, 1))
            plt.imshow(np.array(img), cmap = 'jet', alpha=0.5, clim=(min, max))
            plt.colorbar()
        plt.savefig('./Gen_test_outputs/Grad CAM/MIP/Grad-CAM_gen-rand_' + str(i) + '_norm.png')
        plt.close()

        plt.figure(figsize=(20, 6))
        for ii in range(10):
            plt.subplot(2,5,ii + 1)

            img = Image.fromarray(cam[0,0,128 // 10 * ii,:,:])

            plt.imshow(input_image[0,0,128 // 10 * ii,:,:], cmap='gray', clim=(-1, 1))
            plt.imshow(np.array(img), cmap = 'jet', alpha=0.5)
            plt.colorbar()
        plt.savefig('./Gen_test_outputs/Grad CAM/MIP/Grad-CAM_gen-rand_' + str(i) + '.png')
        plt.close()

        mip_img = torch.mean(input_image, dim=3)
        mip_img = mip_img.cpu().numpy()

        img = (input_image.cpu().numpy() + 1) / 2
        mip_cam = np.true_divide(cam.sum(3),(cam >= 0.01).sum(3))

        plt.imshow(mip_img[0,0,:,:], cmap = 'gray')
        plt.imshow(mip_cam[0,0,:,:], cmap = 'jet', alpha=0.5)
        plt.colorbar()
        plt.savefig('./Gen_test_outputs/Grad CAM/MIP/MIP_Grad-CAM_gen-rand_' + str(i) + '.png')
        plt.close()



        fullcam = grad_cam.generate_fullgradcam(input_image)

        scr = fullcam[1]
        fullcam = fullcam[0]

        min = fullcam.min()
        max = fullcam.max()

        print('\nFullGradCAM:')
        print(scr)

        plt.figure(figsize=(22, 6))
        for ii in range(10):
            plt.subplot(2,5,ii + 1)

            img = fullcam[0,0,128 // 10 // (128 // fullcam.shape[2]) * ii,:,:]

            plt.imshow(input_image[0,0,128 // 10 * ii,:,:], cmap='gray', clim=(-1, 1))
            plt.imshow(img, cmap = 'jet', alpha=0.5)
            plt.colorbar()
        plt.savefig('./Gen_test_outputs/Grad CAM/MIP/Full-Grad-CAM_gen-rand_' + str(i) + '.png')
        plt.close()

        img = (input_image.cpu().numpy() + 1) / 2
        mip_img = np.true_divide(img.sum(3),(img >= 0.01).sum(3))
        mip_cam = np.true_divide(fullcam.sum(3),(fullcam >= 0.01).sum(3))

        plt.imshow(mip_img[0,0,:,:], cmap = 'gray')
        plt.imshow(mip_cam[0,0,:,:], cmap = 'jet', alpha=0.5)
        plt.colorbar()
        plt.savefig('./Gen_test_outputs/Grad CAM/MIP/MIP_Full-Grad-CAM_gen-rand_' + str(i) + '.png')
        plt.close()


        plt.figure(figsize=(22, 6))
        for ii in range(10):
            plt.subplot(2,5,ii + 1)
            plt.imshow(input_image[0,0,128 // 10 * ii,:,:], cmap='gray', clim=(-1, 1))
            plt.colorbar()
        plt.savefig('./Gen_test_outputs/Grad CAM/MIP/Original_gen-rand_' + str(i) + '.png')
        plt.close()

        plt.imshow(mip_img[0,0,:,:], cmap = 'gray')
        plt.colorbar()
        plt.savefig('./Gen_test_outputs/Grad CAM/MIP/Original-mip_gen-rand_' + str(i) + '.png')
        plt.close()
