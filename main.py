import argparse
import pathlib
import time
import math
from os import path, makedirs

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid

from simsiam.loader import TwoCropsTransform
from simsiam.model_factory import SimSiam
from simsiam.criterion import SimSiamLoss
from simsiam.validation import KNNValidation

parser = argparse.ArgumentParser('arguments for training')
parser.add_argument('--data_root', type=str, help='path to dataset directory')
parser.add_argument('--exp_dir', type=str, help='path to experiment directory')
parser.add_argument('--trial', type=str, default='1', help='trial id')
parser.add_argument('--img_dim', default=32, type=int)

parser.add_argument('--arch', default='resnet18', help='model name is used for training')

parser.add_argument('--feat_dim', default=2048, type=int, help='feature dimension')
parser.add_argument('--num_proj_layers', type=int, default=2, help='number of projection layer')
parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
parser.add_argument('--epochs', type=int, default=800, help='number of training epochs')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--loss_version', default='simplified', type=str,
                    choices=['simplified', 'original'],
                    help='do the same thing but simplified version is much faster. ()')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
parser.add_argument('--eval_freq', default=5, type=int, help='evaluate model frequency')
parser.add_argument('--save_freq', default=50, type=int, help='save model frequency')
parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')

parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')


parser.add_argument('--beta', type=float, default=0.25, help='beta for commitment loss')
parser.add_argument('--vq_loss_weight', type=float, default=1.0, help='weight for vq loss')
parser.add_argument('--num_embeddings', type=int, help='size of the codebook')
parser.add_argument('--tau', type=float, default=1.0, help='gumbel-softmax temperature')
parser.add_argument('--embedding_dim', type=int, help='size of the embedding')
parser.add_argument('--fix_tau', action='store_true', default=False, help='Whether to fix gumbel-softmax temperature')
parser.add_argument('--upscale_factor', type=int, default=1, help='Upscale representation')
parser.add_argument('--tau_schedule_end', type=int, default=800, help='Upscale representation')
parser.add_argument('--discrete_type', type=str, choices=['vq', 'gumbel_softmax'], help='vq or gumbel_softmax')
parser.add_argument('--fix_lr', action='store_true', default=False, help='Whether to fix learning rate')
parser.add_argument('--n_proj_conv', type=int, default=2, help='Number of conv layers to use in projector')
parser.add_argument('--strong_proj', action='store_true', default=False, help='Use a strong projector')
parser.add_argument('--res_proj', action='store_true', default=False, help='Use a residual projector')
parser.add_argument('--use_recon', action='store_true', default=False, help='Use reconstruction loss')

args = parser.parse_args()


def main():
    if not path.exists(args.exp_dir):
        makedirs(args.exp_dir)

    def attach_run_id(path, exp_name):
        # From stable-baselines-3
        max_run_id = 0
        path = pathlib.Path(path)
        for dir_path in path.glob(f'{exp_name}_[0-9]*'):
            prefix, _, suffix = dir_path.name.rpartition('_')
            if prefix == exp_name and suffix.isdigit() and int(suffix) > max_run_id:
                max_run_id = int(suffix)
        return f'{exp_name}_{max_run_id + 1}'

    args.trial = attach_run_id(args.exp_dir, args.trial)


    trial_dir = path.join(args.exp_dir, args.trial)
    logger = SummaryWriter(trial_dir)
    print(vars(args))

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_set = datasets.CIFAR10(root=args.data_root,
                                 train=True,
                                 download=True,
                                 transform=TwoCropsTransform(train_transforms))

    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

    model = SimSiam(args)

    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(),
                          # lr=args.learning_rate,
                          # weight_decay=args.weight_decay)

    criterion = SimSiamLoss(args)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)
        cudnn.benchmark = True

    start_epoch = 1
    if args.resume is not None:
        if path.isfile(args.resume):
            start_epoch, model, optimizer = load_checkpoint(model, optimizer, args.resume)
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, start_epoch))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    # routine
    best_acc = 0.0
    validation = KNNValidation(args, model.encoder)
    for epoch in range(start_epoch, args.epochs+1):

        if not args.fix_lr:
            adjust_learning_rate(optimizer, epoch, args)
        if not args.fix_tau:
            adjust_temperature(model, epoch, args)
        print("Training...")

        # train for one epoch
        train_loss, siamese_loss, vq_loss, img, recon = train(train_loader, model, criterion, optimizer, epoch, args)
        logger.add_scalar('Loss/train', train_loss, epoch)
        logger.add_scalar('Loss/siamese_loss', siamese_loss, epoch)
        logger.add_scalar('Loss/vq_loss', vq_loss, epoch)
        if img is not None and recon is not None:
            n = 6
            grid = make_grid(torch.cat((img[:n], recon[:n]), dim=0), nrow=n)
            grid = torch.clamp(grid, min=0.0, max=1.0)
            logger.add_image('train/recon', grid, global_step=epoch)

        if epoch % args.eval_freq == 0:
            print("Validating...")
            val_top1_acc = validation.eval()
            print('Top1: {}'.format(val_top1_acc))

            # save the best model
            if val_top1_acc > best_acc:
                best_acc = val_top1_acc

                save_checkpoint(epoch, model, optimizer, best_acc,
                                path.join(trial_dir, '{}_best.pth'.format(args.trial)),
                                'Saving the best model!')
            logger.add_scalar('Acc/val_top1', val_top1_acc, epoch)

        # save the model
        if epoch % args.save_freq == 0:
            save_checkpoint(epoch, model, optimizer, val_top1_acc,
                            path.join(trial_dir, 'ckpt_epoch_{}_{}.pth'.format(epoch, args.trial)),
                            'Saving...')

    print('Best accuracy:', best_acc)

    # save model
    save_checkpoint(epoch, model, optimizer, val_top1_acc,
                    path.join(trial_dir, '{}_last.pth'.format(args.trial)),
                    'Saving the model at the last epoch.')


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    siamese_losses = AverageMeter('Siamese loss', ':.4e')
    vq_losses = AverageMeter('VQ loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, siamese_losses, vq_losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    img, recon = None, None
    for i, (images, _) in enumerate(train_loader):

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        outs = model(im_aug1=images[0], im_aug2=images[1])
        loss, loss_log = criterion(outs)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # measure elapsed time
        losses.update(loss.item(), images[0].size(0))
        siamese_losses.update(loss_log['siamese_loss'].item(), images[0].size(0))
        vq_losses.update(loss_log['vq_loss'].item(), images[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if 'img1' in loss_log and 'recon1' in loss_log:
            img, recon = loss_log['img1'], loss_log['recon1']

        if i % args.print_freq == 0:
            progress.display(i)
    
    if img is not None and recon is not None:
        img, recon = img.cpu(), recon.cpu()

    return losses.avg, siamese_losses.avg, vq_losses.avg, img, recon


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.learning_rate
    # cosine lr schedule
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_temperature(model, epoch, args):
    """Decay the learning rate based on schedule"""
    # cosine schedule
    tau = args.tau
    if epoch > args.tau_schedule_end:
        tau = 1 / 16
    else:
        tau *= 0.5 * (1. + math.cos(math.pi * epoch / args.tau_schedule_end))
        tau = max(tau, 1 / 16)
    model.encoder.vq_layer.tau = tau 


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(epoch, model, optimizer, acc, filename, msg):
    state = {
        'epoch': epoch,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'top1_acc': acc
    }
    torch.save(state, filename)
    print(msg)


def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename, map_location='cuda:0')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return start_epoch, model, optimizer


if __name__ == '__main__':
    main()



