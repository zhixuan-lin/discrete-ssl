from torch import nn
from torch.nn import functional as F


class SimSiamLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.beta = args.beta
        self.vq_loss_weight = args.vq_loss_weight
        self.args = args
        self.ver = 'simplified'

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



