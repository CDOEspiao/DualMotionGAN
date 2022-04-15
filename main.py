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
NUM_EPOCHS = 1
TRAIN = True
LOAD_WEIGHTS = False
TEST = False
LATEST_CHECKPOINT = False
# CUDA_LAUNCH_BLOCKING = 1
###########################

# Load Data as Numpy Array
MovingMNIST = np.load(r'.data\mnist\raw\mnist_test_seq.npy').transpose(1, 0, 2, 3)
np.random.shuffle(MovingMNIST)

# Train, Test, Validation splits
train_data = MovingMNIST[:8000]
val_data = MovingMNIST[8000:9000]
test_data = MovingMNIST[9000:10000]

train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate)
val_loader = DataLoader(val_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate)

# Save training examples
inp, _ = next(iter(val_loader))
save_training_samples(inp, save_path="examples")

###########################
INPUT_CHANNELS = inp.shape[1]
INPUT_SIZE = (inp.shape[3], inp.shape[4])
ME_CNN_FEATURES = [64, 128, 256]
ME_LSTM_FEATURES = [ME_CNN_FEATURES[-1], ME_CNN_FEATURES[-1], ME_CNN_FEATURES[-1]]
FrameG_FEATURES = [256, 128, 64, 32]
FlowG_FEATURES = [256, 128, 64, 32]
ESTM_FEATURES = [64, 128, 256, 512]
D_FEATURES = [32, 64, 64, 64, 64, 1]
D_POOL_SIZE = [2, 2, 2, 2, 4]
###########################

MotionEncoderCNN = Modules.ProbMotionEncoderCNN(in_channels=INPUT_CHANNELS, features=ME_CNN_FEATURES, kernel_size=3,
                                                stride=1, padding=1, frame_size=INPUT_SIZE).to(device)

MotionEncoderLSTM = Modules.ProbMotionEncoderLSTM(in_channels=ME_CNN_FEATURES[-1], features=ME_LSTM_FEATURES,
                                                  kernel_size=3, padding=1,
                                                  frame_size=(INPUT_SIZE[0]//(2*(len(ME_CNN_FEATURES)-1)),
                                                              INPUT_SIZE[1]//(2*(len(ME_CNN_FEATURES)-1)))).to(device)

FrameGenerator = Modules.Generator(in_channels=ME_LSTM_FEATURES[-1], out_channels=INPUT_CHANNELS,
                                   features=FrameG_FEATURES, kernel_size=3, stride=2, padding=1).to(device)

FrameDiscriminator = Modules.Discriminator(in_channels=INPUT_CHANNELS, features=D_FEATURES, kernel_size=3,
                                           stride=1, padding=1, pool_size=D_POOL_SIZE).to(device)

FlowEstimator = Modules.FlowEstimator(in_channels=2*INPUT_CHANNELS, out_channels=2, features=ESTM_FEATURES,
                                      kernel_size=3, stride=1, padding=1).to(device)

FlowGenerator = Modules.Generator(in_channels=ME_LSTM_FEATURES[-1], out_channels=2, features=FlowG_FEATURES,
                                  kernel_size=3, stride=2, padding=1).to(device)

FlowDiscriminator = Modules.Discriminator(in_channels=2, features=D_FEATURES, kernel_size=3,
                                          stride=1, padding=1, pool_size=D_POOL_SIZE).to(device)

# model = Seq2Seq(in_channels=1, features=[64, 128, 256], kernel_size=3,
#                 stride=1, padding=1, frame_size=(64, 64)).to(device)
# print(colored("Model:", "blue", attrs=["bold"]))
# print(model)

# if LOAD_WEIGHTS:
#     print(colored("Loading latest checkpoint", color='green'))
#     path = latestCheckpoint("checkpoints")
#     model.load_state_dict(torch.load(path)['state_dict'])

real_label = 1.
fake_label = 0.

GFrame_losses = []
DFrame_losses = []
GFlow_losses = []
DFlow_losses = []
EFlow_losses = []

if TRAIN:
    criterion = nn.BCELoss()

    optimizer_me_cnn = torch.optim.Adam(MotionEncoderCNN.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_me_lstm = torch.optim.Adam(MotionEncoderLSTM.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    optimizer_gframe = torch.optim.Adam(FrameGenerator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_dframe = torch.optim.Adam(FrameDiscriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    optimizer_gflow = torch.optim.Adam(FlowGenerator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_dflow = torch.optim.Adam(FlowDiscriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    optimizer_eflow = torch.optim.Adam(FlowEstimator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    print(colored("Starting Training Loop...", "blue"))
    losses = []
    for epoch in range(NUM_EPOCHS + 1):
        print('{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()))
        # if epoch > 0:
        #     checkpoint = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        #     save_checkpoint(checkpoint, "checkpoints/dualMotionGAN_{}.pth.tar".format("".join("%04d" % epoch)))
        train_loss = 0
        MotionEncoderCNN.train()
        MotionEncoderLSTM.train()
        FrameGenerator.train()
        FrameDiscriminator.train()
        FlowGenerator.train()
        FlowDiscriminator.train()
        FlowEstimator.train()

        for batch_num, (inp, target) in enumerate(train_loader, 1):
            out_MotionEncoder = MotionEncoderLSTM(MotionEncoderCNN(inp))
            print("Motion Encoder output: {}".format(out_MotionEncoder.shape))

            # FRAME
            # Frame Discriminator
            FrameDiscriminator.zero_grad()
            real = inp[:, :, 0].to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = FrameDiscriminator(real).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x_real = output.mean().item()

            frame_fake = FrameGenerator(out_MotionEncoder).to(device)
            b_size = frame_fake.size(0)
            label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

            output = FrameDiscriminator(frame_fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_x_fake = output.mean().item()

            errD = errD_real + errD_fake
            optimizer_dframe.step()

            # Frame Generator
            FrameGenerator.zero_grad()
            output = FrameDiscriminator(frame_fake).view(-1)
            label.fill_(real_label)
            errG = criterion(output, label)
            errG.backward(retain_graph=True)
            G_x = output.mean().item()
            optimizer_gframe.step()

            # Use Flow Warping
            fake = FlowGenerator(out_MotionEncoder).to(device)
            fake_frame = torch.zeros((BATCH_SIZE, 1, 64, 64))
            for i in range((inp[:, :, 0]).shape[0]):
                prev = np.array(torch.squeeze(inp[:, :, 0][i].cpu()))
                flow = np.transpose(fake[i].detach().numpy(), (1, 2, 0))
                fake_frame[i] = torch.from_numpy(warp_flow(prev, flow))

            b_size = frame_fake.size(0)
            label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

            output = FrameDiscriminator(fake_frame.detach()).view(-1)
            errD_fake_flowwarp = criterion(output, label)
            errD_fake_flowwarp.backward()
            D_fake_flowwarp = output.mean().item()

            errD = errD_real + errD_fake + errD_fake_flowwarp
            optimizer_dframe.step()

            # FLOW
            # Flow Discriminator
            FlowDiscriminator.zero_grad()
            real = torch.zeros((BATCH_SIZE, 2, 64, 64))
            i = 0
            for prev, next in [torch.squeeze(inp[:, :, 0]), torch.squeeze(inp[:, :, 1])]:
                prev, next = np.array(prev), np.array(next)
                flow = np.transpose(cv2.calcOpticalFlowFarneback(prev, next, True, 0.5, 5, 13, 1, 10, 1.1, 1), (2, 0, 1))
                real[i] = torch.from_numpy(flow)
                i += 1

            b_size = real.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = FlowDiscriminator(real).view(-1)
            errD_flow_real = criterion(output, label)
            errD_flow_real.backward()
            D_flow_real = output.mean().item()

            b_size = fake.size(0)
            label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

            output = FlowDiscriminator(fake.detach()).view(-1)
            errD_flow_fake = criterion(output, label)
            errD_flow_fake.backward()
            D_flow_fake = output.mean().item()

            # Flow Generator
            FlowGenerator.zero_grad()
            output = FlowDiscriminator(fake).view(-1)
            label.fill_(real_label)
            errG_flow = criterion(output, label)
            errG_flow.backward(retain_graph=True)
            G_flow = output.mean().item()
            optimizer_gflow.step()

            # Flow Estimator
            FlowEstimator.zero_grad()
            out_FlowEstimator = FlowEstimator(x=frame_fake, Inp=inp[:, :, 0])
            print("Flow Estimator out: {}".format(out_FlowEstimator.shape))

            b_size = out_FlowEstimator.size(0)
            label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

            output = FlowDiscriminator(out_FlowEstimator.detach()).view(-1)
            errD_flow_fake_estim = criterion(output, label)
            errD_flow_fake_estim.backward()
            D_flow_fake_estim = output.mean().item()

            errD_flow = errD_flow_real + errD_flow_fake + errD_flow_fake_estim
            optimizer_dflow.step()

            # Train Estimator
            output = FlowDiscriminator(out_FlowEstimator).view(-1)
            label.fill_(real_label)
            errE_flow = criterion(output, label)
            errE_flow.backward(retain_graph=True)
            E_flow = output.mean().item()
            optimizer_eflow.step()

            if batch_num % 100 == 0:
                print("Epoch:{} D_x_real:{:.2f} D_x_fake:{:.2f} G_x:{:.2f} D_fake_flowwarp:{:.2f} "
                      "D_flow_real:{:.2f} D_flow_fake:{:.2f} D_flow_fake_estim:{:.2f} E_flow:{:.2f}\n"
                      .format(epoch, D_x_real, D_x_fake, G_x, D_fake_flowwarp,
                              D_flow_real, D_flow_fake, D_flow_fake_estim, E_flow))