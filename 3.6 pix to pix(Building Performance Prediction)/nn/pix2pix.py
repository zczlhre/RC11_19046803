import os
import numpy as np
import time
import datetime
import sys

import torch

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *


epoch=0
n_epochs=100
dataset_name="sunhours"
batch_size=4
lr=0.0002
b1=0.5
b2=0.999
decay_epoch=100
n_cpu=1
img_height=256
img_width=256
channels=3
sample_interval=500
checkpoint_interval=-1

os.makedirs("images/%s" % dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % dataset_name, exist_ok=True)
if __name__=="__main__":

    cuda = True if torch.cuda.is_available() else False

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()

    # Loss weight of L1 pixel-wise loss between translated image and real image
    lambda_pixel = 100

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, img_height // 2 ** 4, img_width // 2 ** 4)

    # Initialize generator and discriminator
    generator = GeneratorUNet()
    discriminator = Discriminator()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()

    if epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (dataset_name, epoch)))
        discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (dataset_name, epoch)))
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    # Configure dataloaders
    transforms_ = [
        transforms.Resize((img_height, img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataloader = DataLoader(
        ImageDataset("../data/%s" % dataset_name, transforms_=transforms_),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
    )

    val_dataloader = DataLoader(
        ImageDataset("../data/%s" % dataset_name, transforms_=transforms_, mode="val"),
        batch_size=10,
        shuffle=True,
        num_workers=1,
    )

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    def sample_images(batches_done):
        """Saves a generated sample from the validation set"""
        imgs = next(iter(val_dataloader))
        real_A = Variable(imgs["B"].type(Tensor))
        real_B = Variable(imgs["A"].type(Tensor))
        fake_B = generator(real_A)
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
        save_image(img_sample, "images/%s/%s.png" % (dataset_name, batches_done), nrow=5, normalize=True)


# ----------
#  Training
# ----------
    prev_time = time.time()

    for epoch in range(epoch, n_epochs):
        for i, batch in enumerate(dataloader):

            # Model inputs
            real_A = Variable(batch["B"].type(Tensor))
            real_B = Variable(batch["A"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # GAN loss
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, valid)
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)

            # Total loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel

            loss_G.backward()

            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_pixel.item(),
                    loss_GAN.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % sample_interval == 0:
                sample_images(batches_done)

        if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (dataset_name, epoch))
            torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (dataset_name, epoch))