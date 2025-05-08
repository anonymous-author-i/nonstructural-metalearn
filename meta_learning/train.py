import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import json

from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        type=str,
        default="MetaMLPTimeEmbed",
        help="choose a model for prediction",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta_mlpt_h2_bl20",
        help="the current run case name",
    )
    parser.add_argument(
        "--future_horizon",
        type=int,
        default=0,
        help="the length of predictive horizon",
    )
    parser.add_argument(
        "--baselearn_len",
        type=int,
        default=20,
        help="the length of historical label-feature for base learner",
    )
    parser.add_argument(
        "--num_epoch", type=int, default=200, help="number of training epochs"
    )
    parser.add_argument(
        "--save_epoch", type=int, default=20, help="save each [x] epochs"
    )
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=200,
        help="minibatch_size used for training",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--reg",
        type=float,
        default=1e-6,
        help="l2-regulation",
    )
    parser.add_argument(
        "--trajectory_clip",
        type=int,
        default=100,
        help="clip the trajectories with limited length < minimal trajectory length",
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default="data/metalearn",
        help="directory of raw simulated data",
    )
    parser.add_argument("--device", default="cuda", help="device used for training")

    opt = parser.parse_args()

    # load setups
    setups = {}
    with open("model_setups.json", "r", encoding="utf-8") as f:
        setups = json.load(f)

    # get model
    model = get_encoder_model(opt, setups, mode="train")

    # get dataset
    train_loader, val_loader, test_loader = get_dataloader(opt, setups, mode="train")

    # training loop
    dir_info = add_dir(opt)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    loss_seq = []

    for epoch in range(1, opt.num_epoch + 1):
        # go through training batches
        model.train()
        train_loss = 0.0
        btch_id = 1
        for batch_inputs, batch_labels in train_loader:
            optimizer.zero_grad()
            loss_, loss_mse = regression_loss(
                model,
                batch_inputs,
                batch_labels,
                param_reg=opt.reg,
                device=opt.device,
            )
            loss_.backward()
            optimizer.step()
            train_loss += loss_mse.item()

            # display batch progress
            progress = btch_id / len(train_loader) * 100
            print(
                f"\rEpoch [{epoch}/{opt.num_epoch}] ",
                "BatchProgress: {0}{1}% ".format(
                    "▉" * int(progress / 10), int(progress)
                ),
                end="",
            )
            btch_id += 1

        # go through val and test batches
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_labels in val_loader:
                loss_, loss_mse = regression_loss(
                    model,
                    batch_inputs,
                    batch_labels,
                    param_reg=opt.reg,
                    device=opt.device,
                )
                val_loss += loss_mse.item()

        test_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_labels in test_loader:
                loss_, loss_mse = regression_loss(
                    model,
                    batch_inputs,
                    batch_labels,
                    param_reg=opt.reg,
                    device=opt.device,
                )
                test_loss += loss_mse.item()

        # reform loss
        if train_loader:
            train_loss /= len(train_loader)
        if val_loader:
            val_loss /= len(val_loader)
        if test_loader:
            test_loss /= len(test_loader)
        loss_seq += [np.array([train_loss, val_loss, test_loss])]

        # display epoch progress
        print(
            f"Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, Test Loss: {test_loss:.4e}"
        )

        # save
        train_log(opt, model, np.vstack(loss_seq), epoch)
