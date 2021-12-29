import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch.distributions as D


def simple_schedule(beta_1, beta_T, T, device):
    betas = torch.linspace(beta_1, beta_T, T, device=device)
    alpha_bars = (1 - betas).cumprod(-1)

    return {"alphas": 1 - betas, "betas": betas, "alpha_bars": alpha_bars}


def cosine_schedule(T, device, s=0.008):
    t = torch.arange(0, T + 1, dtype=torch.float32, device=device)
    f_t = torch.cos(0.5 * np.pi * (t / T + s) / (1 + s)) ** 2

    alpha_bars = f_t / f_t[0]
    betas = (1 - alpha_bars[1:] / alpha_bars[:-1]).clip(min=0, max=0.999)

    return {"alphas": 1 - betas, "betas": betas, "alpha_bars": alpha_bars[1:]}


def forward_process(x_0, t, params):
    # Sample eps from N(0, 1)
    z_t = torch.randn_like(x_0)

    # Index alpha_bar_t
    alpha_bar_t = params["alpha_bars"][t].unsqueeze(-1)

    # Compute x_t as interpolation of x_0 and x_T
    x_t = alpha_bar_t.pow(0.5) * x_0 + (1 - alpha_bar_t).pow(0.5) * z_t

    return z_t, x_t


def reverse_process(z_t, x_t, t, params):
    # Index the diffusion params
    alpha_t, beta_t, alpha_bar_t = [
        params[key][t].unsqueeze(-1) for key in ["alphas", "betas", "alpha_bars"]
    ]

    # Sample noise from N(0, 1)
    eps = torch.randn_like(x_t) if t > 0 else 0

    # Compute the reverse process at step t by removing the predicted noise from x_t
    x_tm1 = (
        alpha_t.pow(-0.5) * (x_t - z_t * beta_t * (1 - alpha_bar_t).pow(-0.5))
        + beta_t * eps
    )

    return x_tm1


def train_loop(data, model, opt, params, args):
    # Sample diffusion time-step
    t = torch.randint(0, args.diffusion_steps, (data.size(0),), device=data.device)

    # Run the forward process
    z_t, x_t = forward_process(data, t, params)

    # Run reverse process network to predict the noise added
    pred_z_t, mu, logs = model(data, x_t, t)
    Q = D.Normal(mu, logs.exp())
    P = D.Normal(torch.zeros_like(mu), torch.ones_like(logs))

    # Compute the simple diffusion loss
    loss_simple = F.mse_loss(pred_z_t, z_t)
    loss_kl = D.kl.kl_divergence(Q, P).mean()

    # Perform backward pass
    opt.zero_grad()
    (loss_simple + 0.25 * loss_kl).backward()
    opt.step()

    return {"diffusion_loss": loss_simple.item(), "kl_loss": loss_kl.item()}


@torch.no_grad()
def test_loop(model, params, args):
    # Start from random noise in N(0, 1)
    x_t = torch.randn(args.n_test_points, args.input_dim, device=args.device)

    # Sample latent from prior
    z_p = torch.randn(x_t.size(0), args.enc_dim, device=args.device)

    # Iteratively run the reverse process to convert noise to data
    for t in tqdm(reversed(range(args.diffusion_steps))):
        tensor_t = torch.tensor(t, device=args.device)
        z_t = model.generate(z_p, x_t, tensor_t)
        x_t = reverse_process(z_t, x_t, t, params)

    return x_t


@torch.no_grad()
def recons_loop(data, model, params, args):
    # Start from random noise in N(0, 1)
    x_t = torch.randn(args.n_test_points, args.input_dim, device=args.device)

    # Sample latent from posterior
    z_q, _, _ = model.posterior(data, temp=0.6)

    # Iteratively run the reverse process to convert noise to data
    for t in tqdm(reversed(range(args.diffusion_steps))):
        tensor_t = torch.tensor(t, device=args.device)
        z_t = model.generate(z_q, x_t, tensor_t)
        x_t = reverse_process(z_t, x_t, t, params)

    return x_t
