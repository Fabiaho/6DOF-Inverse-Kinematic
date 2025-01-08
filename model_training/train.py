from torch.utils.data import DataLoader
import torch
from modules.modules import *
from datasets.dataset_csv import IKDatasetCSV, IKDatasetValCSV
import ikpy.chain
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import json


def train(cfg):
    r_arm = ikpy.chain.Chain.from_urdf_file(cfg.chain_path)

    # gather chain joint limits
    upper = []
    lower = []
    for i in range(1, len(r_arm.links) - 1):
        lower.append(r_arm.links[i].bounds[0])
        upper.append(r_arm.links[i].bounds[1])

    upper = np.array(upper)
    lower = np.array(lower)

    train_dataset = IKDatasetCSV(cfg.train_data_path, with_orientation=True)
    test_dataset = IKDatasetValCSV(cfg.test_data_path, with_orientation=True)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size,
                                 shuffle=False, num_workers=0)

    # build hypernetwork + mainnet
    hypernet = HyperNet(cfg).cuda()
    mainnet = MainNet(cfg).cuda()

    optimizer = torch.optim.Adam(hypernet.parameters(), lr=cfg.lr)

    train_counter, test_counter = 0, 0
    train_loss, test_loss = 0, 0
    best_test_loss = np.inf
    best_test_epoch = 0
    epochs_without_improvements = 0

    train_losses = []
    test_losses = []

    for epoch in range(cfg.num_epochs):
        hypernet.train()

        # -----------------------
        # Training loop
        # -----------------------
        for positions, joint_angles in train_dataloader:
            # positions:  [batch_size, 6]  (x,y,z + Rx,Ry,Rz)
            # joint_angles: [batch_size, 6]
            positions, joint_angles = positions.cuda(), joint_angles.cuda()

            # The MainNet uses an extra column of ones for each sample
            # before concatenating the (true) joint angles
            output = torch.cat(
                (torch.ones(joint_angles.shape[0], 1).cuda(), joint_angles),
                dim=1
            )

            optimizer.zero_grad()

            # predict mixture distribution parameters
            predicted_weights = hypernet(positions)
            distributions, selection = mainnet(output, predicted_weights)

            # compute NLL losses across all mixture components
            losses = [
                -torch.mean(distributions[i].log_prob(
                    joint_angles[:, i].unsqueeze(1)
                ))
                for i in range(len(distributions))
            ]
            loss = sum(losses) / len(losses)

            train_counter += 1
            train_loss += loss.item()

            loss.backward()

            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(hypernet.parameters(), cfg.grad_clip)

            optimizer.step()

        # track average training loss over the epoch
        train_losses.append(train_loss / train_counter)
        print(f"Train loss (Likelihood) {train_losses[-1]}")
        train_loss, train_counter = 0, 0

        # -----------------------
        # Validation loop
        # -----------------------
        sampled = []
        hypernet.eval()

        for test_iter, (positions, joint_angles) in enumerate(test_dataloader):
            positions, joint_angles = positions.cuda(), joint_angles.cuda()
            predicted_weights = hypernet(positions)

            # sample multiple solutions per position
            for j in range(cfg.num_solutions_validation):
                samples, distributions, means, variance, selection = mainnet.validate(
                    torch.ones(joint_angles.shape[0], 1).cuda(),
                    predicted_weights,
                    lower,
                    upper
                )
                sampled.append(samples)

        # compute RMSE on position error
        for sampled_lst in sampled:
            for k in range(len(positions)):
                # construct a full joint vector for forward_kinematics
                # (adding 0 at the start/end if chain indexing requires it)
                joint_angle_vector = [0] + [sampled_lst[i][k].item()
                                            for i in range(cfg.num_joints)] + [0]

                real_frame = r_arm.forward_kinematics(joint_angle_vector)
                # only compare the position part with desired positions
                # (positions[k,:3])
                test_loss += np.sqrt(
                    np.sum(
                        (real_frame[:3, 3]
                         - positions[k, :3].detach().cpu().numpy()) ** 2
                    )
                )
                test_counter += 1

        test_losses.append(test_loss / test_counter)
        print(f"Test loss (RMSE) {test_losses[-1]}")

        # -----------------------
        # Early stopping + checkpoint
        # -----------------------
        if test_losses[-1] < best_test_loss:
            epochs_without_improvements = 0
            best_test_loss = test_losses[-1]
            torch.save(hypernet.state_dict(), f'{cfg.exp_dir}/best_model.pt')
            torch.save(optimizer.state_dict(), f'{cfg.exp_dir}/best_optimizer.pt')
            with open(f'{cfg.exp_dir}/best_test_loss.txt', 'a+') as f:
                f.write(f'Epoch {epoch} - test loss {best_test_loss} \n')
        else:
            epochs_without_improvements += 1

        if epochs_without_improvements == cfg.early_stopping_epochs:
            break

        test_loss, test_counter = 0, 0

        # plotting for quick feedback
        plt.plot(range(len(train_losses)), train_losses, label='train')
        plt.savefig(f'{cfg.exp_dir}/train_plot.png')
        plt.clf()

        plt.plot(range(len(test_losses)), test_losses, label='test')
        plt.savefig(f'{cfg.exp_dir}/test_plot.png')
        plt.clf()

        torch.save(hypernet.state_dict(), f'{cfg.exp_dir}/last_model.pt')
        torch.save(optimizer.state_dict(), f'{cfg.exp_dir}/last_optimizer.pt')


def create_exp_dir(cfg):
    if not os.path.exists(cfg.exp_dir):
        os.mkdir(cfg.exp_dir)
    existing_dirs = os.listdir(cfg.exp_dir)
    if existing_dirs:
        sorted_dirs = sorted(existing_dirs, key=lambda x: int(x.split('_')[1]))
        last_exp_num = int(sorted_dirs[-1].split('_')[1])
        exp_name = f"{cfg.exp_dir}/exp_{last_exp_num + 1}"
    else:
        exp_name = f"{cfg.exp_dir}/exp_0"

    os.makedirs(exp_name)
    with open(f'{exp_name}/run_args.json', 'w+') as f:
        json.dump(cfg.__dict__, f, indent=2)
    return exp_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain-path', type=str,
                        default="assets/digit/urdf/digit_r_arm.urdf",
                        help='URDF chain path')
    parser.add_argument('--train-data-path', type=str,
                        default="data/ur5/ur5_train_data.csv",
                        help='Training data path')
    parser.add_argument('--test-data-path', type=str,
                        default='data/ur5/ur5_test_data.csv',
                        help='Test data path')
    parser.add_argument('--num-joints', type=int, default=6,
                        help='Number of UR5 joints you want to predict')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--num-solutions-validation', type=int, default=10,
                        help='Number of solutions to sample during validation')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Batch size')
    parser.add_argument('--early-stopping-epochs', type=int, default=30,
                        help='Epochs without improvement before stopping')
    parser.add_argument('--grad-clip', type=int, default=1,
                        help='Clip norm of gradient')
    parser.add_argument('--embedding-dim', type=int, default=128,
                        help='Embedding dimension')
    # -----------------------
    #  IMPORTANT:
    #  hypernet-input-dim is now 6 (x,y,z + Rx,Ry,Rz)
    # -----------------------
    parser.add_argument('--hypernet-input-dim', type=int, default=6,
                        help='Number of inputs for the hypernetwork (f)')
    parser.add_argument('--hypernet-hidden-size', type=int, default=1024,
                        help='Hypernetwork hidden layer size')
    parser.add_argument('--hypernet-num-hidden-layers', type=int, default=3,
                        help='Number of hidden layers in hypernetwork')
    parser.add_argument('--jointnet-hidden-size', type=int, default=256,
                        help='JointNet hidden layer size')
    parser.add_argument('--num-gaussians', type=int, default=50,
                        help='Number of Gaussians for mixture. default=1 => no mixture')
    parser.add_argument('--exp_dir', type=str, default='runs',
                        help='Folder path name to save the experiment')

    parser.set_defaults()
    cfg = parser.parse_args()

    full_exp_dir = create_exp_dir(cfg)
    cfg.exp_dir = full_exp_dir

    # The mixture-model output dimension:
    # If num_gaussians=1 => output dimension is 2 (mu, sigma).
    # If num_gaussians>1 => output dimension is num_gaussians * 3 (mu_i, sigma_i, alpha_i)
    cfg.jointnet_output_dim = (cfg.num_gaussians * 2 + cfg.num_gaussians
                               if cfg.num_gaussians != 1 else 2)

    # Launch training
    print(cfg)
    train(cfg)
    