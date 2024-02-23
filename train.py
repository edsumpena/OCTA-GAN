print('Initializing...')

import torch
import os

from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable

import torchio as tio
import nvidia_smi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import OCTA_GAN_128 as model
import gc

from fvcore.nn import FlopCountAnalysis, flop_count_table

gc.collect()
torch.cuda.empty_cache()

BATCH_SIZE = 2
lr_gen = 0.0001
lr_discrim = 0.0002
betas = (0.0, 0.999)
gpu = True
workers = 3
_eps = 1e-15


LAMBDA_DIVERSITY = 0.35

# Latent variable dimension
latent_dim = 1024

loss_func = 'hinge'

ckpts_folder = 'ckpts_our_discrim2'
out_folder = 'out_our_discrim2'
device_indexes = [1]

# Prepare the dataset
data_transforms = {
    'train': tio.Compose([
        tio.RescaleIntensity(percentiles=(0.5, 99.5), out_min_max=(-1, 1)),
    ]),
    'artifact': tio.Compose([
        tio.RescaleIntensity(percentiles=(0.5, 99.5), out_min_max=(-1, 1)),
    ]),
}

aug_affine = tio.RandomAffine(
    scales=0.05,
    degrees=(3, 0, 0),
    translation=(0.15, 0.20, 0.05),
    center='image'
)

aug_contrast = tio.RandomGamma(log_gamma=(-0.20, 0.20))

aug_flip = tio.RandomFlip(axes = ('Left', 'Right'))

augs_train = tio.Compose([
    aug_affine, aug_contrast, aug_flip
])


image_datasets = {
    'train':    # TODO Add training dataloader
}


gpu = "cuda:" + str(device_indexes[0])

device = torch.device(gpu if(torch.cuda.is_available()) else "cpu")
gpu = "Using GPU (" + str(gpu) + ")..."
print(gpu if(torch.cuda.is_available()) else "Using CPU...")

torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    nvidia_smi.nvmlInit()
    print('Number of GPUs: ' + str(nvidia_smi.nvmlDeviceGetCount()))

    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]

    print(available_gpus)

    workers = len(available_gpus) * 4
    workers = 8 if workers > 8 else workers
else:
    workers = 1

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=workers) for x in ['train']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}

def inf_train_gen(data_loader):
    while True:
        for _,images in enumerate(data_loader):
            yield images

G = model.Generator(noise = latent_dim)
D = model.Discriminator(out_class = 1)

g_optimizer = optim.Adam(G.parameters(), lr=lr_gen, betas=betas)
d_optimizer = optim.Adam(D.parameters(), lr=lr_discrim, betas=betas)

print('Gen lr = ' + str(lr_gen))
print('Discrim lr = ' + str(lr_discrim))
print('Betas = ' + str(betas))
print('')
print('Generator:')
print(G)

print('')
print('')

print('Discriminator:')
print(D)

print('')
print('')

inputs = ((torch.randn((1,latent_dim))), (torch.randn((1,1,128,128,128))), (torch.randn((1,1,128,128))))
flops_gen = FlopCountAnalysis(G, inputs[0])
flops_discrim = FlopCountAnalysis(D, inputs[1])


print('Generator Parameters & FLOPS:')
print(flop_count_table(flops_gen))

print('')
print('')

print('Discriminator Parameters & FLOPS:')
print(flop_count_table(flops_discrim))

print('')
print('')
print('')
print('======================== Begin Training ========================')
print('')

G = nn.DataParallel(G, device_ids=device_indexes)
D = nn.DataParallel(D, device_ids=device_indexes)

G.to(device)
D.to(device)

def calc_gradient_penalty(model, x, x_gen, w=10):
    """WGAN-GP gradient penalty"""
    assert x.size()==x_gen.size(), "real and sampled sizes do not match"
    alpha_size = tuple((len(x), *(1,)*(x.dim()-1)))
    alpha_t = torch.cuda.FloatTensor if x.is_cuda else torch.Tensor
    alpha = alpha_t(*alpha_size, device=device).uniform_().to(device)
    x_hat = x.data*alpha + x_gen.data*(1-alpha)
    x_hat = Variable(x_hat, requires_grad=True).to(device)

    def eps_norm(x):
        x = x.view(len(x), -1)
        return (x*x+_eps).sum(-1).sqrt()
    def bi_penalty(x):
        return (x-1)**2

    grad_xhat = torch.autograd.grad(model(x_hat).sum(), x_hat, create_graph=True, only_inputs=True)[0]

    penalty = w*bi_penalty(eps_norm(grad_xhat)).mean()
    return penalty


criterion_l1 = nn.L1Loss()
criterion_mse = nn.MSELoss()

d_fake_losses = list()
d_losses = list()
g_losses = list()
step_counter = list()

c_real = list()
c_rand = list()
c_wasserstein = list()

g_diversity = list()


g_iter = 1
d_iter = 1
cd_iter = 1
TOTAL_ITER = 280000
PRETRAINING_ITER = 0

gen_load = inf_train_gen(dataloaders['train'])
for iteration in range(TOTAL_ITER):
    for p in D.parameters():  
        p.requires_grad = False
    for p in G.parameters():  
        p.requires_grad = True

    ###############################################
    # Train Generator 
    ###############################################
    for iters in range(g_iter):
        G.zero_grad(set_to_none=True)
        real_images = gen_load.__next__()[0]

        _batch_size = real_images.size(0)
        real_images = Variable(real_images).to(device, non_blocking=True)

        z_rand = Variable(torch.randn((_batch_size,latent_dim))).to(device)
        z_rand_2 = Variable(torch.randn((_batch_size,latent_dim))).to(device)

        x_rand = G(z_rand)
        x_rand_2 = G(z_rand_2)

        # Hinge loss component of the generator loss function
        d_fake_loss = D(x_rand).mean()
        d_loss = -d_fake_loss

        # Mode-seeking diversity loss to mitigate mode collapse
        diversity = criterion_l1(x_rand, x_rand_2) / criterion_l1(z_rand, z_rand_2)
        eps = 1 * 1e-5

        diversity_loss = 1 / (diversity + eps)

        loss1 = d_loss + LAMBDA_DIVERSITY * diversity_loss

        if iteration >= PRETRAINING_ITER:
            if iteration == PRETRAINING_ITER and PRETRAINING_ITER != 0:
                print("Discriminator Pretraining Ended. Generator Training Started.")

            if iteration % d_iter == 0:
                if iters<g_iter-1:
                    loss1.backward()
                else:
                    loss1.backward(retain_graph=True)
                g_optimizer.step()
        elif iteration == 0:
            print('Beginning Discriminator Pretraining (' + str(PRETRAINING_ITER) + ' steps)...')


    ###############################################
    # Train D
    ###############################################
    for p in D.parameters():  
        p.requires_grad = True
    for p in G.parameters():  
        p.requires_grad = False

    for iters in range(1):
        d_optimizer.zero_grad(set_to_none=True)
        real_images = gen_load.__next__()[0]

        _batch_size = real_images.size(0)
        z_rand = Variable(torch.randn((_batch_size,latent_dim))).to(device)
        real_images = Variable(real_images).to(device, non_blocking=True)

        x_rand = G(z_rand)

        if loss_func == 'wgan-gp':
            d_real = D(real_images).mean()
            d_rand = D(x_rand).mean()

            x_loss2 = -(d_real - d_rand)

            gradient_penalty_r = calc_gradient_penalty(D,real_images.data, x_rand.data)

            loss2 = x_loss2 + gradient_penalty_r
            loss2.backward(retain_graph=True)
            d_optimizer.step()

        elif loss_func == 'hinge':
            # Compute margin-based hinge loss (positive margin of 1)
            d_real = D(real_images)
            d_real = torch.mean(F.relu(1. - d_real))

            d_rand = D(x_rand)
            d_rand = torch.mean(F.relu(1. + d_rand))
            
            x_loss2 = d_real + d_rand

            loss2 = x_loss2
            loss2.backward(retain_graph=True)
            d_optimizer.step()

    ###############################################
    # Visualization
    ###############################################
    if iteration%500 == 0 or iteration == TOTAL_ITER - 1:
        d_fake_losses.append(d_fake_loss.data.cpu().numpy())#[0])
        d_losses.append(loss2.data.cpu().numpy())#[0])
        g_losses.append(loss1.data.cpu().numpy())

        c_real.append(d_real.data.cpu().numpy())
        c_rand.append(d_rand.data.cpu().numpy())
        c_wasserstein.append(-x_loss2.data.cpu().numpy())

        g_diversity.append(diversity_loss.data.cpu().numpy())

        step_counter.append(TOTAL_ITER if iteration == TOTAL_ITER - 1 else iteration)

        print('[{}/{}]'.format(TOTAL_ITER if iteration == TOTAL_ITER - 1 else iteration,TOTAL_ITER),
              'G: {:<8.3}'.format(loss1.data.cpu().numpy()),
              'D: {:<8.3}'.format(loss2.data.cpu().numpy()),
              'G_rand: {:<8.3}'.format(d_fake_loss.data.cpu().numpy()),
              'G_diversity: {:<8.3}'.format(diversity_loss.data.cpu().numpy()))
        
        print('[ Discriminator Losses ]',
              'D: {:<8.3}'.format(loss2.data.cpu().numpy()), 
              'D_real: {:<8.3}'.format(d_real.data.cpu().numpy()),
              'D_rand: {:<8.3}'.format(d_rand.data.cpu().numpy()),
              'D_wasserstein: {:<8.3}'.format(-x_loss2.data.cpu().numpy()) if loss_func == 'wgan-gp' else 'D_hinge: {:<8.3}'.format(-x_loss2.data.cpu().numpy()))

        plt.figure(figsize=(20,9))
        for i in range(1, 17):
            plt.subplot(2,8,i)
            plt.imshow(x_rand.data.cpu().numpy()[0,0,8 * i - 1,:,:], cmap = 'gray', clim=(-1, 1))
        plt.savefig('./Gen_CKPTs/' + out_folder + '/fake_rand_out_'+str(TOTAL_ITER if iteration == TOTAL_ITER - 1 else iteration)+'.png')
        plt.close()

        plt.figure(figsize=(20,9))
        for i in range(1, 17):
            plt.subplot(2,8,i)
            plt.imshow(real_images.data.cpu().numpy()[0,0,8 * i - 1,:,:], cmap = 'gray', clim=(-1, 1))
        plt.savefig('./Gen_CKPTs/' + out_folder + '/real_out_'+str(TOTAL_ITER if iteration == TOTAL_ITER - 1 else iteration)+'.png')
        plt.close()


        mip_rand = torch.mean(x_rand, dim=3)
        mip_rand = mip_rand.cpu().numpy()

        mip_real = torch.mean(real_images, dim=3)
        mip_real = mip_real.cpu().numpy()

        plt.figure(figsize=[13,10])
        plt.subplot(2,3,1)
        plt.imshow(x_rand.data.cpu().numpy()[0,0,128 // 2,:,:], cmap = 'gray', clim=(-1, 1))

        plt.subplot(2,3,2)
        plt.imshow(real_images.data.cpu().numpy()[0,0,128 // 2,:,:], cmap = 'gray', clim=(-1, 1))

        plt.subplot(2,3,4)
        plt.imshow(mip_rand[0,0,:,:], cmap = 'gray')

        plt.subplot(2,3,5)
        plt.imshow(mip_real[0,0,:,:], cmap = 'gray')

        plt.savefig('./Gen_CKPTs/' + out_folder + '/fake_real_cmp_'+str(TOTAL_ITER if iteration == TOTAL_ITER - 1 else iteration)+'.png')
        plt.close()


        plt.figure(figsize=[11,7.5])
        plt.subplot(2,3,1)
        plt.plot(g_losses, label='G')
        plt.legend()

        plt.subplot(2,3,2)
        plt.plot(d_losses, label='D')
        plt.legend()

        plt.subplot(2,3,3)
        plt.plot(g_losses, label='Gen')
        plt.plot(d_losses, label='Discrim')
        plt.legend()

        plt.subplot(2,3,4)
        plt.plot(d_fake_losses, label='G_rand')
        plt.legend()

        plt.subplot(2,3,5)
        plt.plot(g_diversity, label='G_diversity')
        plt.legend()

        plt.savefig('./Gen_CKPTs/' + out_folder + '/loss_gen_critic_cd_'+str(TOTAL_ITER if iteration == TOTAL_ITER - 1 else iteration)+'.png')
        plt.close()


        plt.figure(figsize=[11,7.5])
        plt.subplot(2,3,1)
        plt.plot(d_losses, label='D')
        plt.legend()

        plt.subplot(2,3,2)
        plt.plot(c_real, label='D_real')
        plt.legend()

        plt.subplot(2,3,3)
        plt.plot(c_rand, label='D_rand')
        plt.legend()

        plt.subplot(2,3,4)
        plt.plot(c_wasserstein, label='D_wasserstein' if loss_func == 'wgan-gp' else 'D_hinge')
        plt.legend()

        plt.savefig('./Gen_CKPTs/' + out_folder + '/loss_critic_'+str(TOTAL_ITER if iteration == TOTAL_ITER - 1 else iteration)+'.png')
        plt.close()


        plt.figure(figsize=[13,7])
        plt.title('Generator and Discriminator Loss', fontsize=24, pad=24)
        plt.xlabel('Steps', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.plot(step_counter, g_losses, label='Generator')
        plt.plot(step_counter, d_losses, label='Discriminator')
        plt.legend(prop={'size': 14})
        plt.savefig('./Gen_CKPTs/' + out_folder + '/loss_curve_'+str(TOTAL_ITER if iteration == TOTAL_ITER - 1 else iteration)+'.png')
        plt.close()

        
    if (iteration+1)%500 == 0 or iteration == 0:
        torch.save(G.state_dict(),'./Gen_CKPTs/' + ckpts_folder + '/G_A_iter'+str(iteration+1)+'.pth')
        torch.save(D.state_dict(),'./Gen_CKPTs/' + ckpts_folder + '/D_A_iter'+str(iteration+1)+'.pth')
