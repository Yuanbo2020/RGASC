import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def move_data_to_gpu(x, cuda):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x


def init_layer(layer):
    """Initialize a Linear or Convolutional layer.
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification." Proceedings of the
    IEEE international conference on computer vision. 2015.
    """

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')

        return x


class Cnn14_asc_aec(nn.Module):
    def __init__(self, classes_num, event_class, batchnormal=False):

        super(Cnn14_asc_aec, self).__init__()

        self.batchnormal=batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)

        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_final = nn.Linear(2048, classes_num, bias=True)

        # self.conv_block1_event = ConvBlock(in_channels=1, out_channels=64)
        # self.conv_block2_event = ConvBlock(in_channels=64, out_channels=128)

        self.conv_block3_event = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4_event = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5_event = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6_event = ConvBlock(in_channels=1024, out_channels=2048)
        self.fc1_event = nn.Linear(2048, 2048, bias=True)
        self.fc_final_event = nn.Linear(2048, event_class, bias=True)

        self.init_weight()

    def init_weight(self):
        if self.batchnormal:
            init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_final)

        init_layer(self.fc1_event)
        init_layer(self.fc_final_event)

    def forward(self, input):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        if self.batchnormal:
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        # x_scene = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        # x_scene = F.dropout(x_scene, p=0.2, training=self.training)
        # x_scene = self.conv_block2(x_scene, pool_size=(2, 2), pool_type='avg')
        # x_scene = F.dropout(x_scene, p=0.2, training=self.training)

        x_scene = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x_scene = F.dropout(x_scene, p=0.2, training=self.training)

        x_scene = self.conv_block4(x_scene, pool_size=(2, 2), pool_type='avg')
        x_scene = F.dropout(x_scene, p=0.2, training=self.training)
        x_scene = self.conv_block5(x_scene, pool_size=(2, 2), pool_type='avg')
        x_scene = F.dropout(x_scene, p=0.2, training=self.training)
        x_scene = self.conv_block6(x_scene, pool_size=(1, 1), pool_type='avg')
        x_scene = F.dropout(x_scene, p=0.2, training=self.training)
        # print(x_scene.size())
        # torch.Size([64, 2048, 10, 2])

        x_scene = torch.mean(x_scene, dim=3)
        (x1_scene, _) = torch.max(x_scene, dim=2)
        x2_scene = torch.mean(x_scene, dim=2)
        x_scene = x1_scene + x2_scene

        x_scene = F.dropout(x_scene, p=0.5, training=self.training)
        x_scene = F.relu_(self.fc1(x_scene))
        scene = self.fc_final(x_scene)


        x_event = self.conv_block3_event(x, pool_size=(2, 2), pool_type='avg')
        x_event = F.dropout(x_event, p=0.2, training=self.training)

        x_event = self.conv_block4_event(x_event, pool_size=(2, 2), pool_type='avg')
        x_event = F.dropout(x_event, p=0.2, training=self.training)
        x_event = self.conv_block5_event(x_event, pool_size=(2, 2), pool_type='avg')
        x_event = F.dropout(x_event, p=0.2, training=self.training)
        x_event = self.conv_block6_event(x_event, pool_size=(1, 1), pool_type='avg')
        x_event = F.dropout(x_event, p=0.2, training=self.training)
        x_event = torch.mean(x_event, dim=3)

        (x1_event, _) = torch.max(x_event, dim=2)
        x2_event = torch.mean(x_event, dim=2)
        x_event = x1_event + x2_event
        x_event = F.dropout(x_event, p=0.5, training=self.training)
        x_event = F.relu_(self.fc1_event(x_event))
        event = torch.sigmoid(self.fc_final_event(x_event))

        return scene, event

