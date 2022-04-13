import matplotlib.pyplot as plt
import glob
import os
import torch
import numpy as np
import torchvision.utils as vutils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def showModelTraining(losses):
    plt.figure(figsize=(10, 5))
    plt.title("Model Loss During Training")
    plt.plot(losses)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def latestCheckpoint(path, mask="*.pth.tar"):
    list_of_files = glob.glob(os.path.join(path, mask))
    latest_file = os.path.basename(max(list_of_files, key=os.path.getctime))
    print("Latest checkpoint: {}".format(latest_file))
    return latest_file


def collate(batch):
    # Add channel dim, scale pixels between 0 and 1, send to GPU
    batch = torch.tensor(np.array(batch)).unsqueeze(1)
    batch = batch / 255.0
    batch = batch.to(device)

    # Randomly pick 10 frames as input, 11th frame is target
    rand = np.random.randint(10,20)
    return batch[:, :, rand-10:rand], batch[:, :, rand]


# Visualize Test model results
def collate_test(batch):
    # Last 10 frames are target
    target = np.array(batch)[:, 10:]

    # Add channel dim, scale pixels between 0 and 1, send to GPU
    batch = torch.tensor(np.array(batch)).unsqueeze(1)
    batch = batch / 255.0
    batch = batch.to(device)
    return batch, target


def createPredImage(batch, output):
    # Prepare plot
    fig, axarr = plt.subplots(batch.shape[0], 11)
    fig.set_size_inches(20, 7)

    for i, sequence in enumerate(batch):
        for video in sequence:
            for j, frame in enumerate(video):
                if j < 10:
                    axarr[i, j].imshow(
                        np.transpose(vutils.make_grid(frame.to('cpu')[:64], padding=2, normalize=True), (1, 2, 0)))
                    axarr[i, j].set_axis_off()
                    if i == 0:
                        axarr[i, j].title.set_text('Input {}'.format(j))

    print("Output: {}".format(output.shape))
    for i, sequence in enumerate(output):
        axarr[i, 10].imshow(np.transpose(vutils.make_grid(sequence[0].to('cpu')[:64], padding=2, normalize=True), (1, 2, 0)))
        axarr[i, 10].set_axis_off()
        if i == 0:
            axarr[i, 10].title.set_text('Prediction')

    plt.show()
