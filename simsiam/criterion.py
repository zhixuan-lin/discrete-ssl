from torch import nn
import torch
from torch.nn import functional as F


class SimSiamLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.beta = args.beta
        self.vq_loss_weight = args.vq_loss_weight
        self.args = args
        self.ver = 'original'
        self.use_recon = args.use_recon

    def asymmetric_loss(self, p, z):
        if self.ver == 'original':
            z = z.detach()  # stop gradient

            p = nn.functional.normalize(p, dim=1)
            z = nn.functional.normalize(z, dim=1)

            return -(p * z).sum(dim=1).mean()

        elif self.ver == 'simplified':
            z = z.detach()  # stop gradient
            return - nn.functional.cosine_similarity(p, z, dim=-1).mean()

    def vq_loss(self, embeddings, encoded):
        codebook_loss = F.mse_loss(embeddings, encoded.detach(), reduction='mean')
        commitment_loss = F.mse_loss(encoded, embeddings.detach(), reduction='mean')
        return codebook_loss, commitment_loss

    def forward(self, outs):

        if self.use_recon:
            img1, img2 = outs['im_aug1'], outs['im_aug2']
            recon1, recon2 = outs['recon1'], outs['recon2']
            indices1, indices2, embeddings1, embeddings2, encoded1, encoded2 = [
                outs[k] for k in (
                    'indices1', 'indices2', 'embeddings1', 'embeddings2', 'encoded1', 'encoded2'
                )]
            codebook_loss1, commitment_loss1 = self.vq_loss(embeddings1, encoded1)
            codebook_loss2, commitment_loss2 = self.vq_loss(embeddings2, encoded2)

            recon_loss1 = F.mse_loss(img1, recon1)
            recon_loss2 = F.mse_loss(img2, recon2)

            recon_loss = 0.5 * recon_loss1 + 0.5 * recon_loss2
            vq_loss = 0.5 * (codebook_loss1 + codebook_loss2 + self.beta * (commitment_loss1 + commitment_loss2))
            loss = recon_loss + self.vq_loss_weight * vq_loss

            # Reconstruction
            mean = torch.as_tensor((0.4914, 0.4822, 0.4465), device=loss.device)[:, None, None]
            std = torch.as_tensor((0.2023, 0.1994, 0.2010), device=loss.device)[:, None, None]
            img1 = img1 * std + mean
            img2 = img2 * std + mean
            recon1 = recon1 * std + mean
            recon2 = recon2 * std + mean

            return loss, {'siamese_loss': recon_loss, 'vq_loss': vq_loss, 'img1': img1, 'recon1': recon1, 'img2': img2, 'recon2': recon2}
        else:
            p1, p2, z1, z2 = (outs[k] for k in ('p1', 'p2', 'z1', 'z2'))
            indices1, indices2, embeddings1, embeddings2, encoded1, encoded2 = [
                outs[k] for k in (
                    'indices1', 'indices2', 'embeddings1', 'embeddings2', 'encoded1', 'encoded2'
                )
            ]
            loss1 = self.asymmetric_loss(p1, z2)
            loss2 = self.asymmetric_loss(p2, z1)
            codebook_loss1, commitment_loss1 = self.vq_loss(embeddings1, encoded1)
            codebook_loss2, commitment_loss2 = self.vq_loss(embeddings2, encoded2)

            siamese_loss = 0.5 * loss1 + 0.5 * loss2
            vq_loss = 0.5 * (codebook_loss1 + codebook_loss2 + self.beta * (commitment_loss1 + commitment_loss2))

            loss = siamese_loss + self.vq_loss_weight * vq_loss
            return loss, {'siamese_loss': siamese_loss, 'vq_loss': vq_loss}



