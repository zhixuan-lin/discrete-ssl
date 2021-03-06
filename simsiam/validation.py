# https://github.com/zhirongw/lemniscate.pytorch/blob/master/test.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch import nn


class KNNValidation(object):
    def __init__(self, args, model, K=1):
        self.model = model
        self.device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')
        self.args = args
        self.K = K

        base_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(root=args.data_root,
                                         train=True,
                                         download=True,
                                         transform=base_transforms)

        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           drop_last=True)

        val_dataset = datasets.CIFAR10(root=args.data_root,
                                       train=False,
                                       download=True,
                                       transform=base_transforms)

        self.val_dataloader = DataLoader(val_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.num_workers,
                                         pin_memory=True,
                                         drop_last=True)
        self.args = args

    def _topk_retrieval(self):
        """Extract features from validation split and search on train split features."""
        n_data = self.train_dataloader.dataset.data.shape[0]
        feat_dim = self.args.feat_dim

        self.model.eval()
        if str(self.device) == 'cuda':
            torch.cuda.empty_cache()

        # train_features = torch.zeros([feat_dim, n_data], device=self.device)
        if self.args.arch == 'resnet18vq':
            train_features = torch.zeros([4 * 4 * self.args.upscale_factor ** 2, n_data], device=self.device)
        else:
            raise Exception()
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.train_dataloader):
                inputs = inputs.to(self.device)
                batch_size = inputs.size(0)

                # forward
                _, _, _, indices = self.model(inputs)
                assert indices.ndim == 3
                features = indices.flatten(start_dim=1)
                # features = nn.functional.normalize(features)
                train_features[:, batch_idx * batch_size:batch_idx * batch_size + batch_size] = features.data.t()

            train_labels = torch.LongTensor(self.train_dataloader.dataset.targets).cuda()

        total = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_dataloader):
                targets = targets.cuda(non_blocking=True)
                batch_size = inputs.size(0)
                _, _, _, indices = self.model(inputs.to(self.device))
                features = indices.flatten(start_dim=1)
                # features = nn.functional.normalize(features)

                # (D, N) -> (1, D, N)
                # (B, D) -> (B, D, 1)
                # hamming distance
                if self.args.upscale_factor < 4:
                    dist = (train_features[None, :, :] == features[:, :, None]).sum(dim=1)
                else:
                    # Split data to avoid OOM issue
                    B = features.size(0)
                    n_chunks = self.args.upscale_factor ** 2 // 4
                    assert B % (n_chunks) == 0
                    chunk_size = B // n_chunks
                    dists = []
                    for i in range(n_chunks):
                        batch_slice = slice(i * chunk_size, (i + 1) * chunk_size)
                        dists.append((train_features[None, :, :] == features[batch_slice, :, None]).sum(dim=1))
                    dist = torch.cat(dists, dim=0)

                # dist = torch.mm(features, train_features)
                yd, yi = dist.topk(self.K, dim=1, largest=True, sorted=True)
                candidates = train_labels.view(1, -1).expand(batch_size, -1)
                retrieval = torch.gather(candidates, 1, yi)

                retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)

                total += targets.size(0)
                correct += retrieval.eq(targets.data).sum().item()
        top1 = correct / total

        return top1

    def eval(self):
        return self._topk_retrieval()


