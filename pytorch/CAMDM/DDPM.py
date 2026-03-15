import json

import torch
import math
import numpy as np


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, scale_betas=1.):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "linear1":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.01
        beta_end = scale * 0.7
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "linear2":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas
        beta_start = scale * 0.01
        beta_end = scale * 0.7
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class DDPM:

    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        # betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        betas = torch.from_numpy(get_named_beta_schedule("cosine", n_steps)).to(torch.float32).to(device)
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        alpha_prev = torch.empty_like(alpha_bars)
        alpha_prev[1:] = alpha_bars[0:n_steps - 1]
        alpha_prev[0] = 1
        self.coef1 = torch.sqrt(alphas) * (1 - alpha_prev) / (1 - alpha_bars)
        self.coef2 = torch.sqrt(alpha_prev) * self.betas / (1 - alpha_bars)

    def sample_forward(self, x, t, eps=None):
        alpha_bar = self.alpha_bars[t].reshape(-1, 1)
        if eps is None:
            eps = torch.randn_like(x)
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        return res

    def sample_backward(self,
                        input_size,
                        xf,
                        net,
                        device,
                        simple_var=True
                        ):
        x = torch.randn(input_size).to(device)
        xf = xf.to(device)
        net = net.to(device)
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, xf, t, net, simple_var)

        return x

    def sample_backward_step(self, x_t, xf, t, net, simple_var=True):
        n = x_t.shape[0]
        t_tensor = torch.tensor([t] * n,
                                dtype=torch.long).to(x_t.device)

        x_0 = net(x_t, xf, t_tensor)

        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (
                        1 - self.alpha_bars[t]) * self.betas[t]
                # print(t, var.numpy())
            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)

        # x_0 = torch.clip(x_0, -1, 1)
        mean = self.coef1[t] * x_t + self.coef2[t] * x_0

        x_t = mean + noise

        return x_t


if __name__ == '__main__':
    n_gpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")
    print(device)

    # 使用4步扩散来匹配示例
    ddpm = DDPM(device=device, n_steps=32)

    # 计算复杂方差并确保为float32
    var = []
    for t in range(ddpm.n_steps):
        if t == 0:
            var_t = np.float32(0.0)  # 明确指定为float32
        else:
            var_t = (1 - ddpm.alpha_bars[t - 1]) / (1 - ddpm.alpha_bars[t]) * ddpm.betas[t]
            var_t = np.float32(var_t.item())  # 转换为Python float32
        var.append(var_t)


    # 辅助函数：确保所有数值为float32
    def ensure_float32(data):
        if isinstance(data, torch.Tensor):
            return np.array(data.cpu().numpy(), dtype=np.float32).tolist()
        elif isinstance(data, (list, np.ndarray)):
            return np.array(data, dtype=np.float32).tolist()
        elif isinstance(data, (float, int)):
            return np.float32(data)
        return data


    # 准备要导出的数据，确保所有浮点数为float32
    output_data = {
        "diffusion_steps": int(ddpm.n_steps),
        "coef1": ensure_float32(ddpm.coef1),
        "coef2": ensure_float32(ddpm.coef2),
        "var": ensure_float32(var),
        "simple_var": ensure_float32(ddpm.betas)
    }


    # 自定义JSON编码器处理float32
    class NumpyFloat32Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.float32):
                return float(obj)
            return super().default(obj)


    # 保存为JSON文件，使用自定义编码器
    with open("ddpm_params_32.json", "w") as f:
        json.dump(output_data, f, indent=4, cls=NumpyFloat32Encoder)

    print("Parameters saved to ddpm_params.json")
