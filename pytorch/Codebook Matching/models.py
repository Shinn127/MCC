import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import Modules
import Utility


class CodebookMatching(nn.Module):
    def __init__(self, encoder, estimator, decoder, xNorm, yNorm, codebook_channels, codebook_dim):
        super(CodebookMatching, self).__init__()

        self.Encoder = encoder
        self.Estimator = estimator
        self.Decoder = decoder

        self.XNorm = Parameter(torch.from_numpy(xNorm), requires_grad=False)
        self.YNorm = Parameter(torch.from_numpy(yNorm), requires_grad=False)

        self.C = codebook_channels
        self.D = codebook_dim

    def sample_gumbel(self, tensor, scale, eps=1e-20):
        scale = scale.reshape(-1, 1, 1, 1)  # This is noise scale between 0 and 1
        noise = torch.rand_like(tensor) - 0.5  # This is random noise between -0.5 and 0.5
        samples = scale * noise + 0.5  # This is noise rescaled between 0 and 1 where 0.5 is default for 0 noise
        return -torch.log(-torch.log(samples + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature, scale):
        y = logits + self.sample_gumbel(logits, scale)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, scale):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature, scale)

        y_soft = y.view(logits.shape)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        y_hard = y_hard.view(logits.shape)

        return y_soft, y_hard

    def sample(self, z, knn):
        z = z.reshape(-1, self.C, self.D)
        z = z.unsqueeze(0).repeat(knn.size(0), 1, 1, 1)
        z_soft, z_hard = self.gumbel_softmax(z, 1.0, knn)
        z_soft = z_soft.reshape(-1, self.C * self.D)
        z_hard = z_hard.reshape(-1, self.C * self.D)
        return z_soft, z_hard

    def forward(self, x, knn, t=None):  # x=input, knn=samples, t=output
        # training
        if t is not None:
            # Normalize
            x = Utility.Normalize(x, self.XNorm)
            t = Utility.Normalize(t, self.YNorm)

            # Encode Y
            target_logits = self.Encoder(torch.cat((t, x), dim=1))
            target_probs, target = self.sample(target_logits, knn)

            # Encode X
            estimate_logits = self.Estimator(x)
            estimate_probs, estimate = self.sample(estimate_logits, knn)

            # Decode
            y = self.Decoder(target)

            # Renormalize
            return Utility.Renormalize(y,
                                       self.YNorm), target_logits, target_probs, target, estimate_logits, estimate_probs, estimate

        # inference
        else:
            # Normalize
            x = Utility.Normalize(x, self.XNorm)

            # Encode X
            estimate_logits = self.Estimator(x)
            estimate_probs, estimate = self.sample(estimate_logits, knn)

            # Decode
            y = self.Decoder(estimate)

            # Renormalize
            return Utility.Renormalize(y, self.YNorm), estimate


if __name__ == '__main__':
    input_dim = 84
    output_dim = 291

    encoder_dim = 1024
    estimator_dim = 1024
    decoder_dim = 1024

    codebook_channels = 128
    codebook_dim = 8
    codebook_size = codebook_channels * codebook_dim

    dropout = 0.2

    network = CodebookMatching(
        encoder=Modules.LinearEncoder(input_dim + output_dim, encoder_dim, encoder_dim, codebook_size, dropout),
        estimator=Modules.LinearEncoder(input_dim, estimator_dim, estimator_dim, codebook_size, dropout),
        decoder=Modules.LinearEncoder(codebook_size, decoder_dim, decoder_dim, output_dim, dropout),
        xNorm=np.random.rand(2, 84).astype(np.float32),
        yNorm=np.random.rand(2, 291).astype(np.float32),
        codebook_channels=codebook_channels,
        codebook_dim=codebook_dim
    )
    network.eval()

    xBatch = torch.randn(32, 84)
    pred, _ =network(xBatch, knn=torch.ones(1))
    print(pred.shape)