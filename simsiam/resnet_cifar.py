'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from lib.normalize import Normalize

from torch.autograd import Variable

class VQLayer(nn.Module):
    def __init__(self, in_planes, num_embeddings, embedding_dim, upscale_factor):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-math.sqrt(3 / embedding_dim), math.sqrt(3 / embedding_dim))
        self.conv = nn.Conv2d(in_planes, embedding_dim * upscale_factor ** 2, kernel_size=1, stride=1)
        self.upscale_factor = upscale_factor

    def forward(self, x):
        # (B, D, H, W) -> (B, D, HW, 1)
        encoded = self.conv(x)
        encoded = F.pixel_shuffle(encoded, self.upscale_factor)
        B, D, H, W = encoded.size()
        # (B, D, H*W)
        encoded_flat = encoded.view(B, D, H*W)
        # (1, K, D)
        weight = self.codebook.weight[None, ...]
        # (B, 1, H*W) + (1, K, 1) + (1, K, D) @ (B, D, H*W) = (B, K, H*W)
        squared_dist = (encoded_flat ** 2).sum(dim=1, keepdim=True) + (weight ** 2).sum(dim=2, keepdim=True) - 2 * weight @ encoded_flat

        # (B, K, H*W) -> (B, H, W)
        indices = torch.argmin(squared_dist, dim=1).view(B, H, W)
        # (B, H, W, D)
        embeddings = self.codebook(indices)
        embeddings = embeddings.permute(0, 3, 1, 2)
        assert encoded.size() == embeddings.size() == (B, D, H, W)
        # ST Gradient trick
        out = encoded + (embeddings - encoded).detach()
        return out, embeddings, encoded, indices

class GumbelSoftmaxLayer(nn.Module):
    def __init__(self, in_planes, num_embeddings, embedding_dim, tau, upscale_factor):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-math.sqrt(3 / embedding_dim), math.sqrt(3 / embedding_dim))
        self.conv = nn.Conv2d(in_planes, num_embeddings * upscale_factor ** 2, kernel_size=1, stride=1)
        self.tau = tau
        self.upscale_factor = upscale_factor

    def forward(self, x):
        logits = self.conv(x)
        logits = F.pixel_shuffle(logits, self.upscale_factor)
        B, N, H, W = logits.size()
        # (B, N, H, W)
        samples = F.gumbel_softmax(logits, self.tau, dim=1)
        indices = torch.argmax(samples, dim=1)
        # (B, N, H*W)
        samples_flat = samples.view(B, N, H*W)
        # (D, N) = (N, D).permute(1, 0)
        code_transpose = self.codebook.weight.permute(1, 0)
        assert code_transpose.size() == (self.embedding_dim, self.num_embeddings)

        # (D, N) @ (B, N, H*W)  = (B, D, H*W)
        encoding_flat = code_transpose @ samples_flat
        encoding = encoding_flat.view(B, self.embedding_dim, H, W)
        assert encoding.size() == (B, self.embedding_dim, H, W)
        return encoding, encoding.detach(), encoding.detach(), indices


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, low_dim=128):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, low_dim)
        # self.l2norm = Normalize(2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = self.l2norm(out)
        return out

class ResNetVQ(nn.Module):
    def __init__(self, block, num_blocks, num_classes, num_embeddings, embedding_dim, tau, upscale_factor, discrete_type, eval_mode: bool):
        super(ResNetVQ, self).__init__()
        self.in_planes = 64
        # Previous experiments without specifying embedding_dim all uses 512.

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if discrete_type == 'vq':
            self.vq_layer = VQLayer(512, num_embeddings=num_embeddings, embedding_dim=embedding_dim, upscale_factor=upscale_factor)
        elif discrete_type == 'gumbel_softmax':
            self.vq_layer = GumbelSoftmaxLayer(512, num_embeddings=num_embeddings, embedding_dim=embedding_dim, tau=tau, upscale_factor=upscale_factor)
        else:
            raise ValueError('Unsupported discrete_type')
        self.fc = nn.Linear(upscale_factor ** 2 * 4 * 4 * embedding_dim, num_classes)
        self.eval_mode = eval_mode
        # self.l2norm = Normalize(2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out, embeddings, encoded, indices = self.vq_layer(out)
        if self.eval_mode:
            out = out.flatten(start_dim=1)
            return self.fc(out)
        else:
            return out, embeddings, encoded, indices



def ResNet18(low_dim=128):
    return ResNet(BasicBlock, [2,2,2,2], low_dim)

def ResNet18VQ(num_classes=10, num_embeddings=512, embedding_dim=64, tau=1.0, upscale_factor=1, discrete_type='vq', eval_mode=False):
    return ResNetVQ(BasicBlock, [2,2,2,2], num_classes=num_classes, num_embeddings=num_embeddings, embedding_dim=embedding_dim, tau=tau, upscale_factor=upscale_factor,
                    discrete_type=discrete_type, eval_mode=eval_mode)

def ResNet34(low_dim=128):
    return ResNet(BasicBlock, [3,4,6,3], low_dim)

def ResNet50(low_dim=128):
    return ResNet(Bottleneck, [3,4,6,3], low_dim)

def ResNet101(low_dim=128):
    return ResNet(Bottleneck, [3,4,23,3], low_dim)

def ResNet152(low_dim=128):
    return ResNet(Bottleneck, [3,8,36,3], low_dim)


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()
