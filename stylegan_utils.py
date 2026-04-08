# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 09:36:52 2022

@author: pnegr
"""
import torch
import copy
import numpy as np
import sys, os
sys.path.append('/workspace/encuesta/stylegan2-ada-pytorch')
import legacy
import dnnlib
from pickle import load
import PIL.Image
import torch.nn.functional as F


############################################################################
def getFaceFromW(G, w, device):
    ws0 = torch.FloatTensor(w).to(device)
    synth_image = G.synthesis(ws0, noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    return synth_image
############################################################################
def genPictureFromW(G, w, device, filename, newsize=(300,300)):
    ws0 = torch.FloatTensor(w).to(device)
    synth_image = G.synthesis(ws0, noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').resize(newsize).save(filename)
############################################################################
def loadStyleGANnetwork(network_pkl, device):
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore
    return G

def project_wN(
    wfile,
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 100,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore
        
    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples.cpu().numpy().astype(np.float32)     # [N, L, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, L, C]

    # if os.path.exists(wfile) == True:
    #     print('Existe latente -> hago finetune')
    #     with np.load(wfile) as data:
    #         w_avg = data['w'] + 1e-3 * w_avg

    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }


    # Load VGG16 feature detector.
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='bilinear', align_corners=False )
    target_vgg_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise)
        synth_images = G.synthesis(ws, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='bilinear',align_corners=False )
        # Features for synth images.
        synth_vgg_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist_vgg = (target_vgg_features - synth_vgg_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist_vgg + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        #logprint(f'step {step+1:>4d}/{num_steps}: dist_vgg {dist_vgg:<4.2f}  dist_arc {dist_arc:<4.2f}  loss {float(loss):<5.2f}')
        if step % 20 == 0:
            print('step %d / %d : dist_vgg %4.2f loss %5.2f' % (step, num_steps, dist_vgg, float(loss.item())))
        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_opt.detach()
