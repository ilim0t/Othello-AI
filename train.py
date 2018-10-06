#!/usr/bin/env python

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os

from game import Othello
from A3C_Adam import SharedAdam
import torch.multiprocessing as mp


class Net(nn.Module):
    def __init__(self, n: int=8):
        super(Net, self).__init__()
        self.n = n

        # 方策関数用層構造 pi(a|s)
        self.policy_l1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.policy_l2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.policy_l3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

        # 状態価値関数用層構造 V(s)
        self.value_l1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.value_l2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2)
        self.value_l3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.value_l4 = nn.Linear(512, 1)

        # 初期化
        for layer in [self.policy_l1, self.policy_l2, self.value_l1, self.value_l2]:
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0)

    def forward(self, state):
        policy = F.relu(self.policy_l1(state))
        policy = F.relu(self.policy_l2(policy))
        policy = self.policy_l3(policy)

        values = F.relu(self.value_l1(state))
        values = F.relu(self.value_l2(values))
        values = F.relu(self.value_l3(values))
        values = self.value_l4(values.view(values.size(0), -1))
        return policy, values

    def sample_action(self, state):
        flag = self.training

        self.eval()
        policy_logits, _ = self.forward(state)
        policy_dist = logits_to_dist(policy_logits.view(policy_logits.size(0), -1))
        action = np.zeros(self.n * self.n)
        action[policy_dist.sample()] = 1
        action = action.reshape(self.n, self.n)

        if flag:
            self.train()

        return action


def reward_to_return(reward_buf, gamma, last_val=0):
    return_buf = [0]*len(reward_buf)
    return_val = last_val
    for i in reversed(range(len(reward_buf))):
        return_val = return_val * gamma + reward_buf[i]
        return_buf[i] = return_val
    return return_buf


def logits_to_dist(logits, dim=0):
    prob = F.softmax(logits, dim=dim)
    return torch.distributions.Categorical(prob)


def A3C_train(shared_model: nn.Module, optimizer, counter, n):
    model = Net(n)
    model.train()

    env = Othello(n)
    env.play((4, 6))
    state = env.data

    max_episode_count = 10000

    while counter.value < max_episode_count:
        model.load_state_dict(shared_model.state_dict())

        history = []  # (state, action, reward)
        done = False
        episode_length = 0
        while not done and episode_length < n * n:
            action = model.sample_action(torch.from_numpy(state.astype(np.float32)).unsqueeze(0))
            next_state, reward, done = env.step(action)

            history.append((state, action, reward))
            state = next_state

        states, actions, rewards = zip(*history)
        return_ = reward_to_return(rewards, 0.95, 0)

        policy_logit, value = model(torch.Tensor(states))
        td = torch.Tensor(return_) - value.squeeze(1)
        policy_dist = [logits_to_dist(pl) for pl in policy_logit.view(policy_logit.size(0), -1)]

        advantage = td.detach()
        value_loss = td.pow(2).mean()
        policy_loss = (torch.cat([-pd.log_prob(torch.from_numpy(np.where(a.flatten())[0])) for pd, a in zip(policy_dist, actions)]) * advantage).sum()
        loss = value_loss + policy_loss
        optimizer.zero_grad()
        loss.backward()

        # shared_modelの勾配として設定
        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is not None:
                break
            shared_param._grad = param.grad

        optimizer.step()

        # 終了条件を満たす場合、初期化
        if done:
            with counter.get_lock():
                counter.value += 1
            episode_length = 0
            env = Othello(n)
            state = env.data


def main():
    parser = argparse.ArgumentParser(description='PyTorch Othello')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    board_n = 8
    model = Net(board_n).to(device)
    optimizer = SharedAdam(model.parameters(), lr=0.0001)
    optimizer.share_memory()

    os.environ['OMP_NUM_THREADS'] = '1'  # pytorchのmultithreadとnumpyのmultithreadの競合回避
    os.environ['CUDA_VISIBLE_DEVICES'] = ""  # CPU only
    num_processes = 4

    processes = []
    counter = mp.Value('i', 0)

    for n in range(num_processes):
        p = mp.Process(target=A3C_train, args=(model, optimizer, counter, board_n))
        p.start()
        processes.append(p)

    # Wait end
    for p in processes:
        p.join()

    for p in processes:
        p.terminate()


if __name__ == '__main__':
    main()
