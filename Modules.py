import torch
import torch.nn as nn
import torch.nn.functional as F

from ConvLSTM import ConvLSTM


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pool_size=2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               output_padding=output_padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.deconv(x)


class ProbMotionEncoderDown(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256, 512], kernel_size=3, stride=1, padding=1,
                 frame_size=(256, 256)):
        super(ProbMotionEncoderDown, self).__init__()
        self.out_channels = features[-1]
        self.out_size = frame_size

        for _ in range(len(features)):
            self.out_size = (((self.out_size[0] - kernel_size + 2*padding)//stride + 1) // 2,
                             ((self.out_size[1] - kernel_size + 2*padding) // stride + 1) // 2)

        self.downs = nn.Sequential()
        self.downs.add_module("ConvBlock1", ConvBlock(in_channels=in_channels, out_channels=features[0],
                                                      kernel_size=kernel_size, stride=stride,
                                                      padding=padding))

        for feature in range(1, len(features)):
            self.downs.add_module("ConvBlock{}".format(feature + 1),
                                  ConvBlock(in_channels=features[feature - 1], out_channels=features[feature],
                                            kernel_size=kernel_size, stride=stride, padding=padding))

    def forward(self, x):
        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)
        batch_size, _, seq_len, height, width = x.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len, self.out_size[0],  self.out_size[1], device=device)

        # Unroll over time steps
        for time_step in range(seq_len):
            out = self.downs(x[:, :, time_step])
            output[:, :, time_step] = out
        print("ProbMotionEncoderDowns output: {}".format(output.shape))
        return output


class ProbMotionEncoderLSTM(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256, 512], kernel_size=3, stride=1, padding=1,
                 frame_size=(256, 256)):
        super(ProbMotionEncoderLSTM, self).__init__()
        self.downs = nn.Sequential()

        for i in range(3):
            self.downs.add_module("ConvLSTM{}".format(i+1), ConvLSTM(in_channels=features[-1],
                                                                     out_channels=features[-1],
                                                                     kernel_size=(3, 3), padding=(1, 1),
                                                                     activation="relu",
                                                                     frame_size=(frame_size[0]//8,
                                                                                 frame_size[1]//8)))

    def forward(self, x):
        output = self.downs(x)
        output = nn.Sigmoid()(output[:, :, -1])
        print("ProbMotionEncoderLSTM output: {}".format(output.shape))
        return output


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, features=[512, 256, 128, 64],
                 kernel_size=3, stride=2, padding=1, output_padding=1):
        super(Generator, self).__init__()
        self.gen = nn.Sequential()
        self.gen.add_module("DeconvBlock1", DeconvBlock(in_channels=in_channels, out_channels=features[0],
                                                            kernel_size=kernel_size, stride=1, padding=padding,
                                                            output_padding=0))

        for feature in range(1, len(features)):
            self.gen.add_module("DeconvBlock{}".format(feature+1), DeconvBlock(in_channels=features[feature-1],
                                                                                    out_channels=features[feature],
                                                                                    kernel_size=kernel_size,
                                                                                    stride=stride, padding=padding,
                                                                                    output_padding=output_padding))

        self.dwise_deconv = nn.ConvTranspose3d(in_channels=features[-1], out_channels=out_channels,
                                               kernel_size=kernel_size, stride=1, padding=padding, output_padding=0)
        self.bn_2 = nn.BatchNorm3d(out_channels)

    def forward(self, x, activation="sigmoid"):
        x = self.gen(x)
        if activation == "sigmoid":
            x = F.sigmoid(self.bn_2(self.dwise_deconv(x)))
        elif activation == "tanh":
            x = F.tanh(self.bn_2(self.dwise_deconv(x)))
        return x


class FlowEstimator(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512], kernel_size=3, stride=1, padding=1):
        super(FlowEstimator, self).__init__()
        self.flow_estimator = nn.Sequential(
            nn.Conv3d(in_channels=in_channels*2, out_channels=features[0], kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm3d(features[0]),
            nn.ReLU(inplace=True),
        )

        for feature in range(0, len(features)-1):
            self.flow_estimator.add_module("ConvBlock{}".format(feature),
                                           ConvBlock(in_channels=features[feature],
                                                          out_channels=features[feature+1],
                                                          kernel_size=kernel_size,
                                                          stride=stride, padding=padding))
        i = 0
        for feature in range(len(features), 1, -1):
            i += 1
            self.flow_estimator.add_module("DeconvBlock{}".format(i),
                                           DeconvBlock(in_channels=features[feature-1],
                                                            out_channels=features[feature-2],
                                                            kernel_size=kernel_size,
                                                            stride=stride, padding=padding))

        self.dwise_conv = nn.Conv3d(in_channels=features[0], out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding)
        self.bn_1 = nn.BatchNorm3d(out_channels)

    def forward(self, x, Inp):
        x = torch.cat([x, Inp], dim=1)
        x = self.flow_estimator(x)
        x = nn.ReLU(self.bn_1(self.dwise_conv(x)))
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=[32, 64, 64, 64, 64, 1], kernel_size=3,
                 stride=1, padding=1, pool_size=[2, 2, 2, 4, 2]):
        super(Discriminator, self).__init__()
        self.downs = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=features[0], kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm3d(features[0]),
            nn.ReLU(inplace=True),
        )

        for feature in range(1, len(features)):
            self.downs.add_module("ConvBlock{}".format(feature), ConvBlock(in_channels=features[feature-1],
                                                                           out_channels=features[feature],
                                                                                  kernel_size=kernel_size,
                                                                                  stride=stride, padding=padding,
                                                                                  pool_size=pool_size[feature-1]))

    def forward(self, x):
        return self.disc(x)


class FlowWarpingLayer(nn.Module):
    def __init__(self, input, grid):
        super(FlowWarpingLayer, self).__init__()
        self.BInterpol = F.grid_sample(input=input, grid=grid, mode='bilinear')

    def forward(self, x, grid):
        return self.BInterpol(x, grid)