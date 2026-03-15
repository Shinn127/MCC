import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


# Output-Blended MoE Layer
class ExpertLinear(torch.nn.Module):
    def __init__(self, experts, input_dim, output_dim):
        super(ExpertLinear, self).__init__()

        self.experts = experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = self.weights([experts, input_dim, output_dim])
        self.b = self.bias([experts, 1, output_dim])

    def forward(self, x, weights):
        y = torch.zeros((x.shape[0], self.output_dim), device=x.device, requires_grad=True)
        for i in range(self.experts):
            y = y + weights[:, i].unsqueeze(1) * (x.matmul(self.W[i, :, :]) + self.b[i, :, :])
        return y

    def weights(self, shape):
        alpha_bound = np.sqrt(6.0 / np.prod(shape[-2:]))
        alpha = np.asarray(np.random.uniform(low=-alpha_bound, high=alpha_bound, size=shape), dtype=np.float32)
        return Parameter(torch.from_numpy(alpha), requires_grad=True)

    def bias(self, shape):
        return Parameter(torch.zeros(shape, dtype=torch.float), requires_grad=True)


class GNN(torch.nn.Module):
    def __init__(self, gating_indices, gating_input, gating_hidden, gating_output, main_indices, main_input,
                 main_hidden, main_output, dropout, input_norm, output_norm):
        super(GNN, self).__init__()

        if len(gating_indices) + len(main_indices) != len(input_norm[0]):
            print("Warning: Number of gating features (" + str(len(gating_indices)) + ") and main features (" + str(
                len(main_indices)) + ") are not the same as input features (" + str(len(input_norm[0])) + ").")

        self.gating_indices = gating_indices
        self.main_indices = main_indices

        self.G1 = nn.Linear(gating_input, gating_hidden)
        self.G2 = nn.Linear(gating_hidden, gating_hidden)
        self.G3 = nn.Linear(gating_hidden, gating_output)

        self.E1 = ExpertLinear(gating_output, main_input, main_hidden)
        self.E2 = ExpertLinear(gating_output, main_hidden, main_hidden)
        self.E3 = ExpertLinear(gating_output, main_hidden, main_output)

        self.dropout = dropout
        self.Xnorm = Parameter(torch.from_numpy(input_norm), requires_grad=False)
        self.Ynorm = Parameter(torch.from_numpy(output_norm), requires_grad=False)

    def forward(self, x):
        x = (x - self.Xnorm[0]) / self.Xnorm[1]

        # Gating
        g = x[:, self.gating_indices]

        g = F.dropout(g, self.dropout, training=self.training)
        g = self.G1(g)
        g = F.elu(g)

        g = F.dropout(g, self.dropout, training=self.training)
        g = self.G2(g)
        g = F.elu(g)

        g = F.dropout(g, self.dropout, training=self.training)
        g = self.G3(g)

        w = F.softmax(g, dim=1)

        # Main
        m = x[:, self.main_indices]

        m = F.dropout(m, self.dropout, training=self.training)
        m = self.E1(m, w)
        m = F.elu(m)

        m = F.dropout(m, self.dropout, training=self.training)
        m = self.E2(m, w)
        m = F.elu(m)

        m = F.dropout(m, self.dropout, training=self.training)
        m = self.E3(m, w)

        m = m * self.Ynorm[1] + self.Ynorm[0]

        return m


if __name__ == '__main__':
    a = torch.randn(32, 348 + 10)
    b1 = torch.randn(2, 358).numpy()
    b2 = torch.randn(2, 303+10).numpy()

    gating_indices = torch.tensor([(348 + i) for i in range(10)])
    main_indices = torch.tensor([(0 + i) for i in range(348)])
    gating_hidden = 32
    gating_output = 8
    main_hidden = 512
    main_output = 303 + 10

    model = GNN(gating_indices=gating_indices,
                gating_input=len(gating_indices),
                gating_hidden=gating_hidden,
                gating_output=gating_output,
                main_indices=main_indices,
                main_input=len(main_indices),
                main_hidden=main_hidden,
                main_output=main_output,
                dropout=0.3,
                input_norm=b1,
                output_norm=b2)

    c = model(a)
