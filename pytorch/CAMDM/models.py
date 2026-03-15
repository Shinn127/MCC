import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn


class EmbedStyle(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input.to(torch.long)
        output = self.action_embedding[idx]
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()

        # Assume d_model is an even number for convenience
        assert d_model % 2 == 0

        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)
        pos, two_i = torch.meshgrid(i_seq, j_seq, indexing='ij')
        pe_2i = torch.sin(pos / 10000 ** (two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000 ** (two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(1, max_seq_len, d_model)

        self.register_buffer('pe', pe, False)

    def forward(self, x: torch.Tensor):
        n, seq_len, d_model = x.shape
        pe: torch.Tensor = self.pe
        rescaled_x = x * d_model ** 0.5
        return rescaled_x + pe[:, 0:seq_len, :]


class TimestepEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, t):
        return self.pe[t]


class MotionDiffusion(nn.Module):
    def __init__(self, input_dim, traj_dim, pose_dim, output_dim, num_styles, denoising_steps, x_norm, y_norm,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.2, activation="gelu"
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.traj_dim = traj_dim
        self.pose_dim = pose_dim
        self.output_dim = output_dim
        self.num_styles = num_styles
        self.denoising_steps = denoising_steps
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.Xnorm = Parameter(torch.from_numpy(x_norm), requires_grad=False)
        self.Ynorm = Parameter(torch.from_numpy(y_norm), requires_grad=False)

        self.embed_style = EmbedStyle(num_styles, self.latent_dim)
        self.pos_encoding = PositionalEncoding(d_model=latent_dim, max_seq_len=10)
        self.time_encoding = TimestepEmbedding(d_model=latent_dim, max_len=denoising_steps)

        # traj process
        self.traj_proj = nn.Linear(traj_dim, latent_dim)

        # pose process
        self.pose_proj = nn.Linear(pose_dim, latent_dim)

        # time step process
        time_embed_dim = self.latent_dim
        self.time_embedder = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, latent_dim),
        )

        # noised motion process
        self.motion_proj = nn.Linear(output_dim, latent_dim)

        # output process
        self.out_process = nn.Linear(latent_dim, output_dim)

        # Encoder-Decoder
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation,
                                                          batch_first=True
                                                          )
        self.seqEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)

    def forward(self, x, xf, style_idx, timesteps):
        xf = (xf - self.Xnorm[0]) / self.Xnorm[1]

        x = self.motion_proj(x).unsqueeze(1)
        traj = self.traj_proj(xf[:, :84]).unsqueeze(1)
        pose = self.pose_proj(xf[:, 84:]).unsqueeze(1)
        style = self.embed_style(style_idx).unsqueeze(1)
        t = self.time_embedder(self.time_encoding(timesteps)).unsqueeze(1)

        # print(x.shape, traj.shape, pose.shape, style.shape, t.shape)
        xseq = torch.cat([x, traj, pose, style, t], dim=1)
        xseq = self.pos_encoding(xseq)
        out = self.seqEncoder(xseq)[:, 0]

        return self.out_process(out) * self.Ynorm[1] + self.Ynorm[0]


if __name__ == '__main__':
    model = MotionDiffusion(
        input_dim=372,
        traj_dim=84,
        pose_dim=288,
        output_dim=327,
        num_styles=3,
        denoising_steps=4,
        x_norm=np.random.randn(2, 372).astype(np.float32),
        y_norm=np.random.randn(2, 327).astype(np.float32),
    )

    xf = torch.randn(32, 372)
    xt = torch.randn(32, 327)
    sty_idx = torch.randint(0, 3, (32, ))
    time_steps = torch.randint(0, 4, (32, ))

    y = model(xt, xf, sty_idx, time_steps)
    print(y.shape)