import argparse
import ReplayBuffer
from env_wrappers import NormalizedBoxEnv
from tensorboardX import SummaryWriter
import pytorch_util as ptu
from torch import nn as nn
import numpy as np

np.bool = np.bool_
import copy
from policy import TanhGaussianPolicy
from mlp_networks import ConcatMlp

parser = argparse.ArgumentParser()
from collections import namedtuple
import torch
import torch.optim as optim
import os
import random
import gym
import sys

sys.path.append('../../')
from hand_teleop.env.rl_env.inspire_relocate_env import InspireRelocateRLEnv
from hand_teleop.env.rl_env.inspire_relocate_env_new_reward import InspireRelocateRLEnv as InspireRelocateRLEnvReward


class TorchBatchRLAlgorithm(object):
    def __init__(
            self,
            trainer,
            env,
            render,
            logdir,
            loaddir,
            replay_buffer: ReplayBuffer.Simple_ReplayBuffer,
            writer,
            device,
            batch_size,
            num_epochs,
            num_trains_per_train_loop,
            gamma,
            min_num_steps_before_training=0,
            evaluation_eps_interval=10,
            load=False,
    ):

        self.trainer = trainer
        self.env = env
        self.render = render
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.logdir = logdir
        self.loaddir = loaddir
        self.device = device
        self.num_epochs = num_epochs
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.gamma = gamma
        self.min_num_steps_before_training = min_num_steps_before_training
        self.evaluation_eps_interval = evaluation_eps_interval
        self.load = load
        self.writer = writer

    def train(self):
        best_reward = -9999

        # put the network to device
        for net in self.trainer.networks:
            net.to(self.device)
        # start
        self.total_step = 0
        if self.load == True:
            self.trainer.load(self.loaddir)

        for self.epoch in range(0, self.num_epochs):
            # To record evaluation
            if self.epoch % self.evaluation_eps_interval == 0:
                eval_reward_total = 0
                o = self.env.reset()
                eval_eps_reward = 0
                while 1:
                    a = self.trainer.policy.get_evaluate_action(o)
                    a = a.detach().cpu().numpy()
                    a_apply = np.zeros(env.robot.dof)
                    a_apply[env.total_activate_joint_index] = a
                    next_o, r, terminal, _ = self.env.step(a_apply)
                    if self.render:
                        self.env.render()
                    eval_reward_total += r
                    eval_eps_reward += r
                    if terminal:
                        break
                    o = next_o

                self.writer.add_scalar('Reward/eval_eps_reward', eval_reward_total, global_step=self.total_step)

                if eval_eps_reward >= best_reward:
                    self.trainer.save(self.logdir, 'best')
                    best_reward = eval_eps_reward

                self.trainer.save(self.logdir, 'final')

            eps_reward = 0
            observations = []
            actions = []
            rewards = []
            terminals = []
            next_observations = []
            o = self.env.reset()
            step = 0

            while 1:
                a, _ = self.trainer.policy.get_action(o)
                a_apply = np.zeros(env.robot.dof)
                a_apply[env.total_activate_joint_index] = a
                next_o, r, terminal, _ = self.env.step(a_apply)
                eps_reward += r
                if self.render:
                    self.env.render()
                observations.append(o)
                rewards.append(r)
                terminals.append(terminal)
                actions.append(a)
                next_observations.append(next_o)
                self.total_step += 1
                step += 1
                if terminal:
                    break
                o = next_o
            # put data in to simple replay buffer
            actions = np.array(actions)
            if len(actions.shape) == 1:
                actions = np.expand_dims(actions, 1)
            observations = np.array(observations)
            next_observations = np.array(next_observations)
            rewards = np.array(rewards)
            terminals = np.array(terminals)
            gammas = np.ones(rewards.shape) * self.gamma
            self.replay_buffer.pushes(observations, actions, rewards, next_observations, terminals, gammas)

            # self.writer.add_scalar('Reward/eps_reward', eps_reward, global_step=self.total_step)

            if self.total_step >= self.min_num_steps_before_training:
                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    sample_state_matrix, sample_action_matrix, sample_reward_matrix, \
                    sample_next_state_matrix, sample_done_matrix, sample_gamma_matrix = \
                        self.replay_buffer.sample(self.batch_size)

                    train_data = dict(
                        observations=sample_state_matrix,
                        actions=sample_action_matrix,
                        rewards=sample_reward_matrix,
                        terminals=sample_done_matrix,
                        next_observations=sample_next_state_matrix,
                        gammas=sample_gamma_matrix
                    )
                    self.trainer.update(train_data)

                self.training_mode(False)
            print(self.logdir)
            print(self.epoch)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch):
    return {
        k: ptu.from_numpy(x).float()
        for k, x in _filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }


class SACTrainer(object):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            logdir,
            loaddir,
            writer,
            soft_target_tau=1e-2,
            target_update_period=1,
            policy_lr=1e-3,
            qf_lr=1e-3,
            reward_scale=1.0,
            use_automatic_entropy_tuning=True,
            optimizer_class=optim.Adam,
            render_eval_paths=False,
            target_entropy=None,
    ):

        self._num_train_steps = 0
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.logdir = logdir
        self.writer = writer

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            else:
                self.target_entropy = target_entropy
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0

    def update(self, np_batch):
        self._num_train_steps += 1
        batch = np_to_pytorch_batch(np_batch)

        # compute loss
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        gammas = batch['gammas']

        """
        Policy and Alpha Loss
        """
        dist = self.policy(obs)

        new_obs_actions, _, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(-1)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        policy_loss = (alpha * log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        next_dist = self.policy(next_obs)
        new_next_actions, _, new_log_pi = next_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * gammas * target_q_values

        TD_loss1 = (q1_pred - q_target.detach()).pow(2)
        TD_loss2 = (q2_pred - q_target.detach()).pow(2)

        qf1_loss = TD_loss1.mean()
        qf2_loss = TD_loss2.mean()

        self.writer.add_scalar('Loss/Q1_loss', qf1_loss, global_step=self._num_train_steps)
        self.writer.add_scalar('Loss/Q2_loss', qf2_loss, global_step=self._num_train_steps)
        self.writer.add_scalar('Loss/policy_loss', policy_loss, global_step=self._num_train_steps)

        """
        Update networks
        """
        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        if self._num_train_steps >= 10000:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()
        self._n_train_steps_total += 1
        self.try_update_target_networks()

    def try_update_target_networks(self):
        if self._n_train_steps_total % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(
            self.qf1, self.target_qf1, self.soft_target_tau
        )
        ptu.soft_update_from_to(
            self.qf2, self.target_qf2, self.soft_target_tau
        )

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def save(self, directory, mode):
        if mode == 'best':
            isExists = os.path.exists(os.path.join(directory, 'best'))
            if isExists == False:
                os.mkdir(os.path.join(directory, 'best'))
        elif mode == 'final':
            isExists = os.path.exists(os.path.join(directory, 'final'))
            if isExists == False:
                os.mkdir(os.path.join(directory, 'final'))
        torch.save(self.policy.state_dict(), os.path.join(directory, mode, 'policy.pth'))
        torch.save(self.qf1.state_dict(), os.path.join(directory, mode, 'qf1.pth'))
        torch.save(self.qf2.state_dict(), os.path.join(directory, mode, 'qf2.pth'))
        torch.save(self.target_qf1.state_dict(), os.path.join(directory, mode, 'target_qf1.pth'))
        torch.save(self.target_qf2.state_dict(), os.path.join(directory, mode, 'target_qf2.pth'))

    def load(self, directory):
        self.policy.load_state_dict(torch.load(os.path.join(directory, 'policy.pth')))
        self.qf1.load_state_dict(torch.load(os.path.join(directory, 'qf1.pth')))
        self.qf2.load_state_dict(torch.load(os.path.join(directory, 'qf2.pth')))
        self.target_qf1.load_state_dict(torch.load(os.path.join(directory, 'target_qf1.pth')))
        self.target_qf2.load_state_dict(torch.load(os.path.join(directory, 'target_qf2.pth')))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


if __name__ == "__main__":
    seed = 2333
    logdir = './EX2/SAC_{}'.format(seed)
    # if you want to load model and continue to learn, set load=True
    loaddir = './EX/SAC_/best'
    # loaddir = None
    variant = dict(
        layer_size=256,
        replay_buffer_size=int(1E6),
        render=False,
        logdir=logdir,
        loaddir=loaddir,
        batch_size=256,
        num_epochs=600000,
        num_trains_per_train_loop=64,
        min_num_steps_before_training=1000,
        load=False,
        writer=SummaryWriter(logdir),
        gamma=0.99,
        seed=seed
    )
    torch.manual_seed(variant['seed'])
    np.random.seed(variant['seed'])
    torch.manual_seed(variant['seed'])
    torch.cuda.manual_seed(variant['seed'])
    random.seed(variant['seed'])
    torch.cuda.manual_seed_all(variant['seed'])
    torch.backends.cudnn.deterministic = True
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)

    env = InspireRelocateRLEnv(use_gui=False, rotation_reward_weight=0,
                               robot_name="inspire_hand_free",
                               object_name="mustard_bottle", frame_skip=10, use_visual_obs=False)

    # InspireRelocateRLEnvReward
    obs_dim = env.observation_space.low.size
    # action_dim = env.action_space.low.size
    action_dim = env.real_dof

    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )

    replay_buffer = ReplayBuffer.Simple_ReplayBuffer(
        variant['replay_buffer_size'],
        obs_dim,
        action_dim
    )
    trainer = SACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        logdir=variant['logdir'],
        loaddir=variant['loaddir'],
        writer=variant['writer'],
        soft_target_tau=5e-3,
        target_update_period=1,
        policy_lr=3E-4,
        qf_lr=3E-4,
        reward_scale=1,
        use_automatic_entropy_tuning=True,
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        env=env,
        render=variant['render'],
        logdir=variant['logdir'],
        loaddir=variant['loaddir'],
        replay_buffer=replay_buffer,
        writer=variant['writer'],
        device=ptu.device,
        batch_size=variant['batch_size'],
        num_epochs=variant['num_epochs'],
        num_trains_per_train_loop=variant['num_trains_per_train_loop'],
        min_num_steps_before_training=variant['min_num_steps_before_training'],
        load=variant['load'],
        gamma=variant['gamma']
    )

    algorithm.train()
