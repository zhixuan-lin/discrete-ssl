from torch import nn
import torch
from .resnet_cifar import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet18VQ, ResNet18VQShallow, BasicBlock
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

class projection_ConvStrong(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            BasicBlock(in_dim, in_dim),
            BasicBlock(in_dim, in_dim),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
        )

    def forward(self, x):
        x = self.net(x)
        x = F.avg_pool2d(x, x.size(2))
        x = x.view(x.size(0), -1)

        return x

class projection_Conv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            # IMPORTANT
        )

    def forward(self, x):
        x = self.net(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)

class projection_ConvMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            # nn.Conv2d(in_dim, out_dim, 3, 1, 1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, out_dim),
            # IMPORTANT
            nn.BatchNorm1d(out_dim, affine=False),
        )

    def forward(self, x):
        x = self.net(x)
        x = F.avg_pool2d(x, x.size(2))
        x = x.view(x.size(0), -1)
        # NOTE: there is a batchnorm here.
        x = self.mlp(x)

        return x
class projection_ConvMLPRaw(nn.Module):
    def __init__(self, in_dim, out_dim, proj_block_num):
        super().__init__()
        if proj_block_num == 2:
            self.net = nn.Sequential(
                BasicBlock(in_dim, 512, stride=2),
                BasicBlock(512, 512),
            )
        elif proj_block_num == 1:
            self.net = nn.Sequential(
                BasicBlock(in_dim, 512, stride=2),
            )
        else:
            assert proj_block_num == 0
            self.net = nn.Sequential(
                nn.Conv2d(in_dim, 512, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )

        self.mlp = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_dim),
            nn.Linear(out_dim, out_dim),
            # IMPORTANT
            nn.BatchNorm1d(out_dim, affine=False),
        )

    def forward(self, x):
        x = self.net(x)
        x = F.avg_pool2d(x, x.size(2))
        x = x.view(x.size(0), -1)
        # NOTE: there is a batchnorm here.
        x = self.mlp(x)

        return x
class prediction_Conv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
        )

    def forward(self, x):
        x = self.net(x)

        return x

class prediction_ConvStrong(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            BasicBlock(in_dim, in_dim),
            BasicBlock(in_dim, in_dim),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
        )

    def forward(self, x):
        x = self.net(x)

        return x



class SimSiam(nn.Module):
    def __init__(self, args):
        super(SimSiam, self).__init__()
        self.args = args
        # self.backbone = SimSiam.get_backbone(args.arch)
        if args.arch == 'resnet18vq':
            self.backbone = ResNet18VQ(args.num_embeddings, args.tau, reset_prob=args.reset_prob, embedding_dim=args.embedding_dim, upscale_factor=args.upscale_factor, vq=args.vq)
        elif args.arch == 'resnet18vqshallow':
            self.backbone = ResNet18VQShallow(args.num_embeddings, args.tau, reset_prob=args.reset_prob, embedding_dim=args.embedding_dim, upscale_factor=args.upscale_factor, vq=args.vq)
        else:
            raise Exception()
        # self.backbone = ResNet18VQ(args.num_embeddings, args.tau, reset_prob=args.reset_prob)
        # out_dim = self.backbone.fc.weight.shape[1]
        # self.backbone.fc = nn.Identity()

        # self.projector = projection_MLP(out_dim, args.feat_dim,
                                        # args.num_proj_layers)

        # self.encoder = nn.Sequential(
            # self.backbone,
            # self.projector
        # )
        if not args.use_mseloss:
            self.encoder = self.backbone

            # embedding_dim -> num_embeddings
            if args.strong_pred:
                self.predictor = prediction_ConvStrong(args.embedding_dim, args.num_embeddings)
            else:
                self.predictor = prediction_Conv(args.embedding_dim, args.num_embeddings)
        else:
            self.encoder = self.backbone
            if not args.conv_mlp_proj:
                if args.strong_proj:
                    self.projector = projection_ConvStrong(args.embedding_dim, args.feat_dim)
                else:
                    self.projector = projection_Conv(args.embedding_dim, args.feat_dim)
                self.predictor = prediction_MLP(in_dim=args.feat_dim)
            elif args.raw:
                self.projector = projection_ConvMLPRaw(args.embedding_dim, args.feat_dim, args.proj_block_num)
                self.predictor = prediction_MLP(in_dim=args.feat_dim)
            else:
                self.projector = projection_ConvMLP(args.embedding_dim, args.feat_dim)
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

        if not self.args.use_mseloss:
            out1, embeddings1, encoded1, indices1 = self.encoder(im_aug1)
            out2, embeddings2, encoded2, indices2 = self.encoder(im_aug2)
            # z2 = self.encoder(im_aug2)

            logits1 = self.predictor(out1)
            logits2 = self.predictor(out2)

            return {'out1': out1, 'out2': out2, 'logits1': logits1, 'logits2': logits2, 
                    'embeddings1': embeddings1, 'encoded1': encoded1, 'indices1': indices1,
                    'embeddings2': embeddings2, 'encoded2': encoded2, 'indices2': indices2
                    }
        else:
            out1, embeddings1, encoded1, indices1 = self.encoder(im_aug1)
            out2, embeddings2, encoded2, indices2 = self.encoder(im_aug2)
            z1 = self.projector(out1)
            z2 = self.projector(out2)
            p1 = self.predictor(z1)
            p2 = self.predictor(z2)
            return {'p1': p1, 'p2': p2, 'z1': z1, 'z2': z2,
                    'embeddings1': embeddings1, 'encoded1': encoded1, 'indices1': indices1,
                    'embeddings2': embeddings2, 'encoded2': encoded2, 'indices2': indices2}








