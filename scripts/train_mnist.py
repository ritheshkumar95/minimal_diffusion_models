import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time
import yaml
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
import torchvision.transforms as tf
from torch.utils.tensorboard.writer import SummaryWriter

from modules.network import MnistNet
from modules.common import simple_schedule, cosine_schedule, train_loop, test_loop


def make_figure(x):
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    x = make_grid(x, nrow=int(x.size(0) ** 0.5))
    ax.imshow(x.permute(1, 2, 0))
    return fig


def make_infinite_iterator(loader):
    while True:
        for data in loader:
            yield data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=True)

    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--n_downsample", type=int, default=2)
    parser.add_argument("--n_resblocks", type=int, default=5)
    parser.add_argument("--ngf", type=int, default=16)

    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--beta_1", type=int, default=1e-4)
    parser.add_argument("--beta_T", type=int, default=0.02)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--schedule", type=str, default="simple")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--iters", type=int, default=100000)
    parser.add_argument("--n_test_points", type=int, default=16)
    parser.add_argument("--print_interval", type=int, default=50)
    parser.add_argument("--plot_interval", type=int, default=250)
    parser.add_argument("--save_interval", type=int, default=1000)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # set the seed
    seed = 111
    np.random.seed(seed)
    torch.manual_seed(seed)

    root = Path(args.save_path)
    root.mkdir(parents=True, exist_ok=True)
    (root / "imgs").mkdir(exist_ok=True)

    #######################
    # Create data loaders #
    #######################
    train_loader = torch.utils.data.DataLoader(
        MNIST(
            "./files/",
            train=True,
            download=True,
            transform=tf.Compose(
                [
                    tf.ToTensor(),
                    tf.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    data_itr = make_infinite_iterator(train_loader)

    test_loader = torch.utils.data.DataLoader(
        MNIST(
            "./files/",
            train=False,
            download=True,
            transform=tf.Compose(
                [
                    tf.ToTensor(),
                    tf.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=args.n_test_points,
        shuffle=True,
    )

    ####################################
    # Dump arguments and create logger #
    ####################################
    writer = SummaryWriter(str(root))
    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)

    #########################
    # Create PyTorch Models #
    #########################
    model = MnistNet(
        args.input_dim,
        args.n_downsample,
        args.n_resblocks,
        args.ngf,
        args.diffusion_steps,
    ).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    print(model)

    ######################
    # Dump Original Data #
    ######################
    orig_data = next(iter(test_loader))[0]
    fig = make_figure(orig_data)
    writer.add_figure("original", fig, 0)

    ###############################
    # Create diffusion parameters #
    ###############################
    if args.schedule == "simple":
        params = simple_schedule(
            args.beta_1, args.beta_T, args.diffusion_steps, args.device
        )
    elif args.schedule == "cosine":
        params = cosine_schedule(args.diffusion_steps, args.device)

    ################################

    costs = {
        "diffusion_loss": [],
    }
    start = time.time()
    for iters in range(args.iters):
        model.train()

        # Sampe data
        data = next(data_itr)[0].to(args.device)

        # Run train loop
        metrics = train_loop(data, model, opt, params, args)

        # Update tensorboard
        for key, val in metrics.items():
            writer.add_scalar(key, val, iters)
            costs[key].append(val)

        if iters % args.print_interval == 0:
            mean_costs = [f"{key}: {np.mean(val):.3f}" for key, val in costs.items()]
            log = (
                f"Steps {iters} | "
                f"ms/batch {1e3 * (time.time() - start) / args.print_interval:5.2f} | "
                f"loss {mean_costs}"
            )
            print(log)

            costs = {key: [] for key in costs.keys()}
            start = time.time()

        if iters % args.plot_interval == 0:
            model.eval()
            st = time.time()
            print("#" * 30)
            print("Generating samples")

            with torch.no_grad():
                samples = test_loop(orig_data.shape, model, params, args)

            fig = make_figure(samples)
            writer.add_figure("generated", fig, iters)
            fig.savefig(root / "imgs" / ("sample_%05d.png" % iters))

            print(f"Completed in {time.time() - st:.2f}")
            print("#" * 30)

        if iters % args.save_interval == 0:
            torch.save(model.state_dict(), root / "model.pt")


if __name__ == "__main__":
    main()
