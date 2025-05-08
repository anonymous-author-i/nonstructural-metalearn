import numpy as np
import os
import json
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from models.encoders import *


def create_segments(opt, settings, mode):
    """
    Create segments: Batch trajectories -> Trajectory segments for learning
    """
    if mode == "train":
        baselearn_len=opt.baselearn_len
        future_horizon=opt.future_horizon
    elif mode == "test":
        baselearn_len=settings["baselearn_len"]
        future_horizon=settings["future_horizon"]
    
    raw_data_dir = opt.raw_data_dir

    history_len = settings["history_len"]
    state_select = settings["state_select"]
    traj_clip = opt.trajectory_clip
    try:
        control_select = settings["control_select"]
    except KeyError:
        control_select = []
    output_select = settings["output_select"]

    segment_len = baselearn_len + history_len + 1 + future_horizon

    # get trajectories and create batch loader
    traj_dir_list = [file for file in os.listdir(raw_data_dir) if file.endswith(".npz")]
    print("{} trajectories loaded.".format(len(traj_dir_list)))

    input_seg_loader = []
    output_seg_loader = []

    for traj_dir in traj_dir_list:
        # load and select dimensions
        data_ = np.load(raw_data_dir + "/" + traj_dir)
        x_seq_batch = data_["x_seq_batch"][state_select, :, :]
        u_seq_batch = data_["u_seq_batch"][control_select, :, :]
        dx_real_seq_batch = data_["dx_real_seq_batch"][output_select, :, :]
        dx_nomi_seq_batch = data_["dx_nomi_seq_batch"][output_select, :, :]

        if control_select:
            input_batch_seq = np.concatenate([x_seq_batch, u_seq_batch], axis=0)
        else:
            input_batch_seq = x_seq_batch
        output_batch_seq = dx_real_seq_batch - dx_nomi_seq_batch
        input_batch_seq = input_batch_seq.transpose(2, 1, 0)
        output_batch_seq = output_batch_seq.transpose(2, 1, 0)
        ## permute (batch_size, traj_len, vec_dim)

        # segments_len should be shorter than trajectory length in any simulated batches
        assert segment_len <= input_batch_seq.shape[1]

        # go through all batches
        for ii in range(input_batch_seq.shape[0]):
            # clip trajectory
            input_seq = input_batch_seq[ii, :traj_clip, :]
            output_seq = output_batch_seq[ii, :traj_clip, :]
            ## (traj_len, vec_dim)

            # split trajectory into samller segemnts
            left = input_seq.shape[0] % segment_len
            seg_num = input_seq.shape[0] // segment_len
            input_seg_list = np.array_split(
                input_seq[: input_seq.shape[0] - left, :], seg_num, axis=0
            )
            output_seg_list = np.array_split(
                output_seq[: input_seq.shape[0] - left, :], seg_num, axis=0
            )

            # add to seg_loader
            input_seg_loader += input_seg_list
            output_seg_loader += output_seg_list

    seg_wrapper = list(zip(input_seg_loader, output_seg_loader))
    print("{} segments created.".format(len(seg_wrapper)))

    # shuffle segments
    np.random.shuffle(seg_wrapper)
    input_seg_loader = [tup[0] for tup in seg_wrapper]
    output_seg_loader = [tup[1] for tup in seg_wrapper]

    # loader to dataset
    X_dataset = np.dstack(input_seg_loader).transpose(2, 0, 1)
    Y_dataset = np.dstack(output_seg_loader).transpose(2, 0, 1)
    X_dataset = torch.tensor(X_dataset, dtype=torch.float32).to(opt.device)
    Y_dataset = torch.tensor(Y_dataset, dtype=torch.float32).to(opt.device)
    ## (segment_size, traj_len, vec_dim)

    # add output to X for meta_learning
    # normal learning also supported by removing output labels from inputs
    X_dataset = torch.concat([X_dataset, Y_dataset], dim=2)

    # taylor the Y_dataset since we only need current + future outputs for loss computation
    Y_dataset = Y_dataset[:, -future_horizon - 1 :, :]
    print(X_dataset.shape, Y_dataset.shape)

    return X_dataset, Y_dataset


def make_loader(X_dataset, Y_dataset, opt, split_ratio):
    minibatch_size = opt.minibatch_size
    device = opt.device

    dataset = TensorDataset(X_dataset, Y_dataset)
    dataset_len = len(X_dataset)
    train_size = int(dataset_len * split_ratio[0] / 10)
    val_size = int(dataset_len * split_ratio[1] / 10)
    test_size = dataset_len - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=minibatch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=minibatch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def add_dir(opt):
    log_path = "log/" + opt.model_name
    learned_models_path = "learned_models/" + opt.model_name
    os.mkdir(log_path)
    os.mkdir(learned_models_path)
    # copy model setups to dir
    # shutil.copyfile("model_setups.json", learned_models_path + "/model_setups.json")
    setups = {}
    with open("model_setups.json", "r", encoding="utf-8") as f:
        setups = json.load(f)
        with open(
            "{}/model_setup.json".format(learned_models_path), "w", encoding="utf-8"
        ) as file:
            current_model_setup = {key: value for key, value in setups.items() if key == opt.encoder}
            # add new info into dict
            extra_info = {"baselearn_len": opt.baselearn_len, "future_horizon": opt.future_horizon}
            current_model_setup[opt.encoder].update(extra_info)
            json.dump(current_model_setup, file)
    # save training params
    f = open(learned_models_path + "/train_params.txt", "w")
    f.write(
        "encoder: "
        + str(opt.encoder)
        + "\n"
        + "num_epoch: "
        + str(opt.num_epoch)
        + "\n"
        + "minibatch_size: "
        + str(opt.minibatch_size)
        + "\n"
        + "lr: "
        + str(opt.lr)
        + "\n"
        + "reg: "
        + str(opt.reg)
        + "\n"
        + "trajectory_clip: "
        + str(opt.trajectory_clip)
        + "\n"
    )


def train_log(opt, model, loss_seq, epoch):
    log_path = "log/" + opt.model_name
    learned_models_path = "learned_models/" + opt.model_name
    np.save(log_path + "/loss_record.npy", loss_seq)
    if not epoch % opt.save_epoch or epoch == 1 or epoch == opt.num_epoch:
        torch.save(
            model.state_dict(),
            log_path + "/model_param_epoch{}.pth".format(epoch),
        )

    torch.save(
        model.state_dict(),
        learned_models_path + "/model_param.pth",
    )


def test_log(opt, model):
    pass


def regression_loss(model, inputs, labels, param_reg, device, reduction="mean"):
    # get outputs to device
    outputs = model(inputs).to(device)
    # define Loss
    loss_obj = torch.nn.MSELoss(reduction=reduction)
    loss_mse = loss_obj(outputs, labels).to(device)
    # define L2 loss
    l2_reg = torch.tensor(0.0, requires_grad=True).to(device)
    for param in model.parameters():
        l2_reg = l2_reg + torch.norm(param, 2)
    loss = loss_mse + param_reg * l2_reg
    return loss, loss_mse


def get_dataloader(opt, setups, mode, split_ratio=[8, 1, 1]):
    settings = setups[opt.encoder]
    X_dataset, Y_dataset = create_segments(opt, settings, mode)
    train_loader, val_loader, test_loader = make_loader(
        X_dataset, Y_dataset, opt, split_ratio
    )

    return train_loader, val_loader, test_loader


def get_encoder_model(opt, setups, mode):
    settings = setups[opt.encoder]
    if mode == "train":
        baselearn_len=opt.baselearn_len
        future_horizon=opt.future_horizon
    elif mode == "test":
        baselearn_len=settings["baselearn_len"]
        future_horizon=settings["future_horizon"]
    else:
        raise AttributeError("Only 'train' and 'test' mode are supported")
    if opt.encoder == "MLP":
        model = MLP_Encoder(
            state_dim=len(settings["state_select"]),
            control_dim=len(settings["control_select"]),
            output_dim=len(settings["output_select"]),
            hidden_sizes=settings["hidden_sizes"],
            dropout=settings["drop_out"],
            device=opt.device,
        )
        model.to(opt.device)

    elif opt.encoder == "MLPTimeEmbed":
        model = MLPTimeEmbed_Encoder(
            state_dim=len(settings["state_select"]),
            control_dim=len(settings["control_select"]),
            output_dim=len(settings["output_select"]),
            history_len=settings["history_len"],
            hidden_sizes=settings["hidden_sizes"],
            dropout=settings["drop_out"],
            device=opt.device,
        )
        model.to(opt.device)

    elif opt.encoder == "MetaMLP":
        model = MetaMLP_Encoder(
            state_dim=len(settings["state_select"]),
            control_dim=len(settings["control_select"]),
            output_dim=len(settings["output_select"]),
            base_reg=settings["base_reg"],
            baselearn_len=baselearn_len,
            future_horizon=future_horizon,
            hidden_sizes=settings["hidden_sizes"],
            dropout=settings["drop_out"],
            device=opt.device,
        )
        model.to(opt.device)

    elif opt.encoder == "MetaMLPTimeEmbed":
        model = MetaMLPTimeEmbed_Encoder(
            state_dim=len(settings["state_select"]),
            control_dim=len(settings["control_select"]),
            output_dim=len(settings["output_select"]),
            base_reg=settings["base_reg"],
            history_len=settings["history_len"],
            baselearn_len=baselearn_len,
            future_horizon=future_horizon,
            hidden_sizes=settings["hidden_sizes"],
            dropout=settings["drop_out"],
            device=opt.device,
        )
        model.to(opt.device)
    
    elif opt.encoder == "MetaMLPtMLP":
        model = MetaMLPtMLP_Encoder(
            state_dim=len(settings["state_select"]),
            control_dim=len(settings["control_select"]),
            output_dim=len(settings["output_select"]),
            base_reg=settings["base_reg"],
            history_len=settings["history_len"],
            baselearn_len=baselearn_len,
            future_horizon=future_horizon,
            mlpt_encoder_sizes=settings["mlpt_encoder_sizes"],
            mlpt_encoded_dim=settings["mlpt_encoded_dim"],
            hidden_sizes=settings["hidden_sizes"],
            dropout=settings["drop_out"],
            device=opt.device,
        )
        model.to(opt.device)

    params = model.parameters()

    # count params
    num_params = 0
    for param in params:
        num_params += torch.prod(torch.tensor(param.size()))

    print("total params: ", num_params)

    return model
