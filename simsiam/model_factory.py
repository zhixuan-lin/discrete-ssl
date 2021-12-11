from torch import nn
import torch
from .resnet_cifar import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet18VQ, BasicBlock
import torch.nn.functional as F



# class projection_MLP(nn.Module):
#     def __init__(self, in_dim, out_dim, num_layers=2):
#         super().__init__()
#         hidden_dim = out_dim
#         self.num_layers = num_layers

#         self.layer1 = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True)
#         )

#         self.layer2 = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True)
#         )
#         self.layer3 = nn.Sequential(
#             nn.Linear(hidden_dim, out_dim),
#             nn.BatchNorm1d(out_dim, affine=False)  # Page:5, Paragraph:2
#         )

    # def forward(self, x):
    #     if self.num_layers == 2:
    #         x = self.layer1(x)
    #         x = self.layer3(x)
    #     elif self.num_layers == 3:
    #         x = self.layer1(x)
    #         x = self.layer2(x)
    #         x = self.layer3(x)

    #     return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()
        out_dim = in_dim
        hidden_dim = int(out_dim / 4)

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x


class projection_ConvFC(nn.Module):
    def __init__(self, in_dim, out_dim, n_proj_conv, hidden_dim=256):
        super().__init__()
        module_list = []
        last_dim = in_dim
        for i in range(n_proj_conv):
            module_list.append(nn.Conv2d(last_dim, hidden_dim, 3, 1, 1))
            module_list.append(nn.BatchNorm2d(hidden_dim))
            module_list.append(nn.ReLU(inplace=True))
            last_dim = hidden_dim
        module_list.append(nn.Conv2d(hidden_dim, hidden_dim, 4, 2, 1))
        module_list.append(nn.BatchNorm2d(hidden_dim))
        module_list.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*module_list)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False),
        )

    def forward(self, x):
        x = self.net(x)
        x = F.avg_pool2d(x, x.size(2))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class SimSiam(nn.Module):
    def __init__(self, args):
        super(SimSiam, self).__init__()
        self.args = args
        if args.arch == 'resnet18vq':
            self.backbone = ResNet18VQ(
                num_classes=10,
                num_embeddings=args.num_embeddings,
                embedding_dim=args.embedding_dim,
                tau=args.tau,
                upscale_factor=args.upscale_factor,
                discrete_type=args.discrete_type,
                eval_mode=False
            )
        else:
            raise ValueError('Only resnet18vq supported')

        self.encoder = self.backbone
        self.projector = projection_ConvFC(args.embedding_dim, args.feat_dim, args.n_proj_conv)
        self.predictor = prediction_MLP(in_dim=args.feat_dim)

    # @staticmethod
    # def get_backbone(backbone_name):
    #     return {'resnet18': ResNet18(),
    #             'resnet34': ResNet34(),
    #             'resnet50': ResNet50(),
    #             'resnet101': ResNet101(),
    #             'resnet152': ResNet152()}[backbone_name]
    #     return ResNet18VQ()

    def forward(self, im_aug1, im_aug2):

        out1, embeddings1, encoded1, indices1 = self.encoder(im_aug1)
        out2, embeddings2, encoded2, indices2 = self.encoder(im_aug2)
        z1 = self.projector(out1)
        z2 = self.projector(out2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return {'p1': p1, 'p2': p2, 'z1': z1, 'z2': z2,
                'embeddings1': embeddings1, 'encoded1': encoded1, 'indices1': indices1,
                'embeddings2': embeddings2, 'encoded2': encoded2, 'indices2': indices2}








