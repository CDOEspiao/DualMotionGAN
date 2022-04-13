import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from termcolor import colored
from model import Seq2Seq

import imageio
from datetime import datetime
import re
import cv2

import Modules
from Support import *

# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device: " + colored(str(device), color='blue', attrs=['bold']))
print(torch.cuda.get_device_name(0))

###########################
LEARNING_RATE = 0.0001
BATCH_SIZE = 2
NUM_EPOCHS = 2
TRAIN = True
LOAD_WEIGHTS = False
TEST = False
LATEST_CHECKPOINT = False
###########################

# Load Data as Numpy Array
MovingMNIST = np.load(r'C:\Users\psurd\PycharmProjects\DualMotionGAN\.data\mnist\raw\mnist_test_seq.npy').transpose(1, 0, 2, 3)
np.random.shuffle(MovingMNIST)

# Train, Test, Validation splits
train_data = MovingMNIST[:8000]
val_data = MovingMNIST[8000:9000]
test_data = MovingMNIST[9000:10000]


def save_checkpoint(state, path="checkpoints/dualMotionGAN_0000.pth.tar"):
    print(colored("Saving checkpoints {}".format(path), color='green'))
    torch.save(state, path)


train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate)
val_loader = DataLoader(val_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate)

# Save input examples
inp, _ = next(iter(val_loader))
# Reverse process before displaying
inp = inp.cpu().numpy() * 255.0
print("Input: {}".format(inp.shape))
for i, video in enumerate(inp.squeeze(1)[:3]):          # Loop over videos
    path = r"C:\Users\psurd\PycharmProjects\DualMotionGAN\examples\example_{}.gif".format(i)
    imageio.mimsave(path, video.astype(np.uint8), "GIF", fps=5)
    print("Saving example: {}".format(path))


model = Seq2Seq(in_channels=1, features=[64, 128, 256], kernel_size=3,
                stride=1, padding=1, frame_size=(64, 64)).to(device)
print(colored("Model:", "blue", attrs=["bold"]))
print(model)

MotionEncoderDowns = Modules.ProbMotionEncoderDown(in_channels=1, features=[64, 128, 256], kernel_size=3,
                                                   stride=1, padding=1, frame_size=(64, 64)).to(device)
MotionEncoderLSTM = Modules.ProbMotionEncoderLSTM(in_channels=1, features=[64, 128, 256], kernel_size=3,
                                                  stride=1, padding=1, frame_size=(64, 64)).to(device)
#
FrameGenerator = Modules.Generator(in_channels=256, out_channels=1, features=[256, 128, 64],
                                   kernel_size=3, stride=2, padding=1).to(device)
FlowGenerator = Modules.Generator(in_channels=256, out_channels=1, features=[256, 128, 64],
                                   kernel_size=3, stride=2, padding=1).to(device)
FlowEstimator = Modules.FlowEstimator(in_channels=1, out_channels=2, features=[64, 128, 256],
                                           kernel_size=3, stride=1, padding=1).to(device)
FrameDiscriminator = Modules.Discriminator(in_channels=1, features=[32, 64, 64, 64, 1], kernel_size=3,
                                           stride=1, padding=1, pool_size=[2, 2, 2, 2]).to(device)
FlowDiscriminator = Modules.Discriminator(in_channels=2, features=[32, 64, 64, 64, 1], kernel_size=3,
                                          stride=1, padding=1, pool_size=[2, 2, 2, 2]).to(device)

print(colored("Additional blocks", "blue", attrs=["bold"]))
print(colored("Flow Estimator: ", "blue", attrs=["bold"]))
print(FlowEstimator)
print(colored("Frame Discriminator: ", "blue", attrs=["bold"]))
print(FrameDiscriminator)
print(colored("Flow Discriminator: ", "blue", attrs=["bold"]))
print(FlowDiscriminator)

if LOAD_WEIGHTS:
    print(colored("Loading latest checkpoint", color='green'))
    path = latestCheckpoint(r"C:\Users\psurd\PycharmProjects\DualMotionGAN\checkpoints")
    model.load_state_dict(torch.load(path)['state_dict'])

real_label = 1.
fake_label = 0.

GFrame_losses = []
DFrame_losses = []
GFlow_losses = []
DFlow_losses = []

if TRAIN:
    criterion = nn.BCELoss()
    optimizer_dframe = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_dflow = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_gframe = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_gflow = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    print(colored("Starting Training Loop...", "blue"))
    losses = []
    for epoch in range(NUM_EPOCHS + 1):
        print('{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()))
        # if epoch > 0:
        #     checkpoint = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        #     save_checkpoint(checkpoint, "checkpoints/dualMotionGAN_{}.pth.tar".format("".join("%04d" % epoch)))
        train_loss = 0
        model.train()

        for batch_num, (inp, target) in enumerate(train_loader, 1):
            out_MotionEncoder = MotionEncoderLSTM(MotionEncoderDowns(inp))

            # FRAME
            # FrameDiscriminator
            FrameDiscriminator.zero_grad()
            real = inp[0].to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = FrameDiscriminator(real).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x_real = output.mean().item()

            fake = FrameGenerator(out_MotionEncoder).to(device)
            b_size = fake.size(0)
            label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

            output = FrameDiscriminator(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_x_fake = output.mean().item()

            errD = errD_real + errD_fake
            optimizer_dframe.step()

            # FrameGenerator
            FrameGenerator.zero_grad()
            output = FrameDiscriminator(fake).view(-1)
            label.fill_(real_label)
            errG = criterion(output, label)
            errG.backward()
            G_x = output.mean().item()
            optimizer_gframe.step()

            if 1 == 1:
                break


            # FLOW
            # FlowDiscriminator
            FlowDiscriminator.zero_grad()
            real = cv2.calcOpticalFlowFarneback(inp[0], inp[1], True, 0.5, 5, 13, 1, 10, 1.1, 1)

                # to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = FlowDiscriminator(real).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x_real = output.mean().item()

            fake = FlowGenerator(out_MotionEncoder).to(device)
            b_size = fake.size(0)
            label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

            output = FlowDiscriminator(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_x_fake = output.mean().item()

            errD = errD_real + errD_fake
            optimizer_dflow.step()

            # FlowGenerator
            FlowGenerator.zero_grad()
            output = FlowDiscriminator(fake).view(-1)
            label.fill_(real_label)
            errG = criterion(output, label)
            errG.backward()
            G_x = output.mean().item()
            optimizer_gflow.step()



#             output = model(inp)
#             loss = criterion(output.flatten(), target.flatten())
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             train_loss += loss.item()
#
#             # Output training stats
#             if batch_num % 500 == 0:
#                 print('[%d/%d][%d/%d]\tLoss: %.4f'
#                       % (epoch, NUM_EPOCHS, batch_num, len(train_loader), loss.item()))
#
#             # Save Losses for plotting later
#             losses.append(loss.item())
#
#         train_loss /= len(train_loader.dataset)
#
#         val_loss = 0
#         model.eval()
#         with torch.no_grad():
#             for input, target in val_loader:
#                 output = model(input)
#                 loss = criterion(output.flatten(), target.flatten())
#                 val_loss += loss.item()
#         val_loss /= len(val_loader.dataset)
#
#         print("Epoch:{} Training Loss:{:.2f} Validation Loss:{:.2f}\n".format(epoch, train_loss, val_loss))
#
#         if epoch % 5 == 0:
#             showModelTraining(losses)

