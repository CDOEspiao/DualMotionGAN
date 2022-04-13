import torch
import torch.nn as nn
import torch.nn.functional as  F

from Modules import *


class Seq2Seq(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256], kernel_size=3, stride=1, padding=1, frame_size=(64, 64)):
        super(Seq2Seq, self).__init__()

        self.ProbabilisticMotionEncoderDown = ProbMotionEncoderDown(in_channels=in_channels, features=features,
                                                            kernel_size=kernel_size, stride=stride, padding=padding,
                                                            frame_size=frame_size)

        self.ProbabilisticMotionEncoderLSTM = ProbMotionEncoderLSTM(in_channels=in_channels, features=features,
                                                                    kernel_size=kernel_size, stride=stride,
                                                                    padding=padding,
                                                                    frame_size=frame_size)

        self.Generator = Generator(in_channels=256, out_channels=1, features=[256, 128, 64],
                                   kernel_size=3, stride=2, padding=1)

        self.FlowEstimator = FlowEstimator(in_channels=1, out_channels=2, features=[64, 128, 256],
                                           kernel_size=3, stride=1, padding=1)

        self.FusingLayer = nn.Conv2d(in_channels=in_channels*2, out_channels=1, kernel_size=kernel_size,
                                     stride=stride, padding=padding)

    def forward(self, X):
        y = self.ProbabilisticMotionEncoderLSTM(self.ProbabilisticMotionEncoderDown(X))
        futureFrameGenerator = self.Generator(y, activation="sigmoid")                  # I..(t+1)
        futureFlowGenerator = self.Generator(y, activation="tanh")                      # F_(t+1)
        y_2 = F.grid_sample(input=y[-1], grid=futureFlowGenerator, mode='bilinear')     # I_(t+1)
        result = nn.Sigmoid()(self.FusingLayer(torch.cat([futureFrameGenerator, y_2])))
        return result