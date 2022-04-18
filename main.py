import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from termcolor import colored
from model import DualMotionGAN

import imageio
from datetime import datetime
import re
import cv2
import random

import Modules
from Support import *

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: " + colored(str(device), color="blue", attrs=["bold"]))
print(torch.cuda.get_device_name(0))

###########################
LEARNING_RATE = 0.001
BATCH_SIZE = 2
NUM_EPOCHS = 5
TRAIN = True
SAVE_CHECKPOINTS = False
LOAD_WEIGHTS = False
TEST = False
LATEST_CHECKPOINT = False
SAVE_TRAINING_SAMPLES = False
SAVE_EPOCHS_OUTPUT = True
###########################

# Load Data as Numpy Array
MovingMNIST = np.load(r".data\mnist\raw\mnist_test_seq.npy").transpose(1, 0, 2, 3)
np.random.shuffle(MovingMNIST)

# Train, Test, Validation splits
train_data = MovingMNIST[:8000]
val_data = MovingMNIST[8000:9000]
test_data = MovingMNIST[9000:10000]

train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate)
val_loader = DataLoader(val_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate)

# Save training examples
inp, _ = next(iter(val_loader))
if SAVE_TRAINING_SAMPLES:
    save_training_samples(inp, save_path="examples")

###########################
INPUT_CHANNELS = inp.shape[1]
INPUT_SIZE = (inp.shape[3], inp.shape[4])
ME_CNN_FEATURES = (64, 128, 256)
ME_LSTM_FEATURES = (ME_CNN_FEATURES[-1], ME_CNN_FEATURES[-1], ME_CNN_FEATURES[-1])
FrameG_FEATURES = (256, 128, 64)
FlowG_FEATURES = (256, 128, 64)
D_FEATURES = (32, 64, 128, 256)
###########################

model = DualMotionGAN(in_channels=INPUT_CHANNELS, frame_size=INPUT_SIZE, device=device,
                      me_cnn_features=ME_CNN_FEATURES, me_lstm_features=ME_LSTM_FEATURES,
                      gframe_features=FrameG_FEATURES, gflow_features=FlowG_FEATURES).to(device)

if LOAD_WEIGHTS:
    print(colored("Loading latest checkpoint", color='green'))
    path = latestCheckpoint("checkpoints")
    model.load_state_dict(torch.load(path)['model_state_dict'])

if TRAIN:
    FrameDiscriminator = Modules.Discriminator(in_channels=INPUT_CHANNELS, features=D_FEATURES, kernel_size=4, stride=2,
                                               padding=1, bias=False).to(device)
    FlowDiscriminator = Modules.Discriminator(in_channels=2, features=D_FEATURES, kernel_size=4, stride=2, padding=1,
                                              bias=False).to(device)
    criterion = nn.L1Loss()
    criterion_2 = nn.BCELoss()
    optimizer_model = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_dframe = torch.optim.Adam(FrameDiscriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_dflow = torch.optim.Adam(FlowDiscriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    if LOAD_WEIGHTS:
        FrameDiscriminator.load_state_dict(torch.load(path)['dframe_state_dict'])
        FlowDiscriminator.load_state_dict(torch.load(path)['dflow_state_dict'])
        optimizer_model.load_state_dict(torch.load(path)['model_optimizer'])
        optimizer_dframe.load_state_dict(torch.load(path)['dframe_optimizer'])
        optimizer_dflow.load_state_dict(torch.load(path)['dflow_optimizer'])

    real_label = 1.
    fake_label = 0.

    model_losses, DFrame_losses, DFlow_losses = [], [], []
    model.train()
    FrameDiscriminator.train()
    FlowDiscriminator.train()

    print(colored("Starting Training Loop...", "magenta"))

    for epoch in range(NUM_EPOCHS):
        print('\n{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()))
        if SAVE_CHECKPOINTS:
            if epoch > 0:
                checkpoint = {'epoch': epoch,
                              'model_state_dict': model.state_dict(), 'model_optimizer': optimizer_model.state_dict(),
                              'dframe_state_dict': model.state_dict(), 'dframe_optimizer': optimizer_dframe.state_dict(),
                              'dflow_state_dict': model.state_dict(), 'dflow_optimizer': optimizer_dflow.state_dict()}
                save_checkpoint(checkpoint, "checkpoints/dualMotionGAN_{}.pth.tar".format("".join("%04d" % epoch)))

        train_loss = 0

        for batch_num, (inp, target) in enumerate(train_loader, 1):
            model.zero_grad()
            FrameDiscriminator.zero_grad()
            FlowDiscriminator.zero_grad()

            frame_prediction, flow_prediction, prediction = model(inp)
            if batch_num == 1:
                if SAVE_EPOCHS_OUTPUT:
                    with torch.no_grad():
                        save_output(save_dir="examples", epoch=epoch, inp=inp, target=target, prediction=prediction,
                                    flow_prediction=flow_prediction, frame_prediction=frame_prediction)

            # FRAME
            # 1.Frame Discriminator
            # 1.1.Real
            real = inp[:, :, random.randint(0, inp.shape[2]-1)].to(device)
            b_size = real.size(0)

            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = FrameDiscriminator(real).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x_real = output.mean().item()

            # 1.2.Frame Generator Output
            label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
            output = FrameDiscriminator(frame_prediction.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_x_fake = output.mean().item()

            # 1.3.Prediction
            label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
            output = FrameDiscriminator(prediction.detach()).view(-1)
            errD_fake_prediction = criterion(output, label)
            errD_fake_prediction.backward()
            D_fake_prediction = output.mean().item()

            errD = errD_real + errD_fake + errD_fake_prediction
            optimizer_dframe.step()

            # FLOW
            # 2.Flow Discriminator
            # 2.1.Real
            real = torch.zeros((b_size, 2, INPUT_SIZE[0], INPUT_SIZE[1])).to(device)
            frame_example = random.randint(0, inp.shape[2]-2)
            i = 0
            for prev, next in [torch.squeeze(inp[:, :, frame_example]), torch.squeeze(inp[:, :, frame_example+1])]:
                prev, next = np.array(prev.cpu()), np.array(next.cpu())
                flow = np.transpose(cv2.calcOpticalFlowFarneback(prev, next, True, 0.5, 5, 13, 1, 10, 1.1, 1),
                                    (2, 0, 1))
                real[i] = torch.from_numpy(flow)
                i += 1

            b_size = real.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = FlowDiscriminator(real).view(-1)
            errD_flow_real = criterion(output, label)
            errD_flow_real.backward()
            D_flow_real = output.mean().item()

            # 2.2.Flow Generator output
            label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
            output = FlowDiscriminator(flow_prediction.detach()).view(-1)
            errD_flow_fake = criterion(output, label)
            errD_flow_fake.backward()
            D_flow_fake = output.mean().item()

            errD_flow = errD_flow_real + errD_flow_fake
            optimizer_dflow.step()

            # 3.Frame Generator
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = FrameDiscriminator(frame_prediction).view(-1)
            errG_x_disc = criterion(output, label)
            errG_x_disc.backward(retain_graph=True)

            err_FG = criterion(frame_prediction, target)
            err_FG.backward(retain_graph=True)

            err_FrameG = errG_x_disc + err_FG

            # 4.Flow Generator
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = FlowDiscriminator(flow_prediction).view(-1)
            errG_flow_disc = criterion(output, label)
            errG_flow_disc.backward(retain_graph=True)
            G_flow_disc = output.mean().item()

            real_flow = torch.zeros((b_size, 2, INPUT_SIZE[0], INPUT_SIZE[1])).to(device)
            i = 0
            for prev, next in [torch.squeeze(inp[:, :, -1]), torch.squeeze(target)]:
                prev, next = np.array(prev.cpu()), np.array(next.cpu())
                flow = np.transpose(cv2.calcOpticalFlowFarneback(prev, next, True, 0.5, 5, 13, 1, 10, 1.1, 1), (2, 0, 1))
                real_flow[i] = torch.from_numpy(flow)
                i += 1

            errG_flow = criterion(flow_prediction, real_flow)
            errG_flow.backward(retain_graph=True)
            err_FlowG = errG_flow_disc + errG_flow

            # 5.Prediction
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = FrameDiscriminator(prediction).view(-1)
            err_disc = criterion(output, label)
            err_disc.backward(retain_graph=True)
            E_disc = output.mean().item()

            err_model = criterion(prediction, target)
            err_model.backward()
            optimizer_model.step()

            if batch_num % 200 == 0:
                print("\nEpoch:{}\tIteration:{}/{}\n"
                      "DFrame Real: {:.3f} DFrame FrameG: {:.3f} DFrame Prediction: {:.3f} Error: {:.3f}\n"
                      "DFlow Real: {:.3f} DFlow FlowG: {:.3f} Error: {:.3f}\n"
                      "Frame Generator Error - D: {:.3f}, Error (by D + Target Frame): {:.3f}\n"
                      "Flow Generator - D:  {:.3f}, Error (by D + 10-11 Flow): {:.3f}\n"
                      "Model - D: {:.3f}, Error (by D + Target Frame): {:.3f}"
                      .format(epoch, batch_num, len(train_loader),
                              D_x_real, D_x_fake, D_fake_prediction, errD.item(),
                              D_flow_real, D_flow_fake, errD_flow.item(),
                              errG_x_disc, err_FrameG,
                              G_flow_disc, err_FlowG,
                              E_disc, err_model.item()))

            # # Save Losses for plotting later
            # G_losses.append(errG.item())
            # D_losses.append(errD.item())
            #
            # model_losses
            # DFrame_losses
            # DFlow_losses = [], [], [], []
