# -*- coding: utf-8 -*-
import gc
import logging
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger('DDPG')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# GPU usage
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WEIGHT_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4

def fan_in_uniform_init(tensor, fan_in=None):
    # Utility function for initializing actor and critic
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1.0 / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1.0 - tau) * target_param.data + tau * param.data)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size[0])

        # Layer 2
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_size[1])

        # Output layer
        self.output = nn.Linear(hidden_size[1], num_outputs)

        # Weight initialization
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)
        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)
        nn.init.uniform_(self.output.weight, -WEIGHT_FINAL_INIT, WEIGHT_FINAL_INIT)
        nn.init.uniform_(self.output.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs):
        x = inputs
        # Layer 1
        x = self.linear1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # Layer 2
        x = self.linear2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # Output layer
        outputs = torch.tanh(self.output(x))
        return outputs

class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.bn1 = nn.BatchNorm1d(hidden_size[0])

        # Layer 2
        self.linear2 = nn.Linear(hidden_size[0] + num_outputs, hidden_size[1])
        self.bn2 = nn.BatchNorm1d(hidden_size[1])

        # Output layer (single value)
        self.V = nn.Linear(hidden_size[1], 1)

        # Weight initialization
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)
        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)
        nn.init.uniform_(self.V.weight, -WEIGHT_FINAL_INIT, WEIGHT_FINAL_INIT)
        nn.init.uniform_(self.V.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs, actions):
        x = inputs
        # Layer 1
        x = self.linear1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # Layer 2
        x = torch.cat((x, actions), 1) # Insert the actions
        x = self.linear2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # Output
        V = self.V(x)
        return V

class DDPG(object):
    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space, checkpoint_dir=None):
        """
        Deep Deterministic Policy Gradient (https://arxiv.org/abs/1509.02971)
        :param gamma:           Discount factor
        :param tau:             Update factor for the actor and the critic
        :param hidden_size:     Number of units in the hidden layers of the actor and critic. (must be of length 2)
        :param num_inputs:      Size of the input states
        :param action_space:    The action space of the used environment.
                                Used to clip the actions and to distinguish the number of outputs
        :param checkpoint_dir:  Path as String to the directory to save the networks
                                If None then "./save_models/" will be used
        """
        self.gamma = gamma
        self.tau = tau
        self.action_space = action_space

        # Define the actor
        self.actor = Actor(hidden_size, num_inputs, self.action_space).to(dev)
        self.actor_target = Actor(hidden_size, num_inputs, self.action_space).to(dev)

        # Define the critic
        self.critic = Critic(hidden_size, num_inputs, self.action_space).to(dev)
        self.critic_target = Critic(hidden_size, num_inputs, self.action_space).to(dev)

        # Define the optimizers for both networks
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=1e-2)

        # Make sure both targets are with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # Set the directory to save the models
        if checkpoint_dir is None:
            self.checkpoint_dir = "./saved_models/"
        else:
            self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info('Saving all checkpoints to {}'.format(self.checkpoint_dir))

    def calc_action(self, state, action_noise=None):
        """
        Evaluates the action to perform in a given state
        :param state:           State to perform the action on in the env.
                                Used to evaluate the action.
        :param action_noise:    If not None, the noise to apply on the evaluated action
        :return:
        """
        x = state.to(dev)

        # Get the continuous action value to perform in the env
        self.actor.eval()
        mu = self.actor(x)
        self.actor.train()
        mu = mu.data

        # During training we add noise for exploration
        if action_noise is not None:
            noise = torch.Tensor(action_noise.noise()).to(dev)
            mu += noise

        # Clip the output according to the action space of the env
        mu = mu.clamp(self.action_space.low[0], self.action_space.high[0])

        return mu

    def update_params(self, batch):
        """
        Updates the parameters/networks of the agent according to the given batch.
            1. Compute the target
            2. Update the Q-function/critic by one step of gradient descent
            3. Update the policy/actor by one step of gradient ascent
            4. Update the target networks through a soft update
        :param batch:   Batch to perform the training of the parameters
        :return:
        """
        # Get tensors from the batch
        state_batch = torch.cat(batch.state).to(dev)
        action_batch = torch.cat(batch.action).to(dev)
        reward_batch = torch.cat(batch.reward).to(dev)
        done_batch = torch.cat(batch.done).to(dev)
        next_state_batch = torch.cat(batch.next_state).to(dev)

        # Get the actions and the state values to compute the targets
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch.detach())

        # Compute the target
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)
        expected_values = reward_batch + (1.0 - done_batch) * self.gamma * next_state_action_values

        # Clipping the expected values here?
        # expected_values = torch.clamp(expected_values, min_value, max_value)

        # Update the critic networks
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optimizer.step()

        # Update the actor networks
        self.actor_optimizer.zero_grad()
        policy_loss = self.critic(state_batch, self.actor(state_batch))
        policy_loss = -policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def save_checkpoint(self, last_timestep, replay_buffer):
        """
        Saving the networks and all parameters to a file in 'checkpoint_dir'
        :param last_timestep:  Last timestep in training before saving
        :param replay_buffer:   Current replay buffer
        :return:
        """
        checkpoint_name = self.checkpoint_dir + '/ep_{}.pth.tar'.format(last_timestep)
        logger.info('Saving checkpoint...')
        checkpoint = {
            'last_timestep': last_timestep,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'replay_buffer': replay_buffer
        }
        logger.info('Saving model at timestep {}...'.format(last_timestep))
        torch.save(checkpoint, checkpoint_name)
        gc.collect()
        logger.info('Saved model at timestep {} to {}'.format(last_timestep, self.checkpoint_dir))

    def get_path_of_latest_file(self):
        """
        Returns the latest created file in 'checkpoint_dir'
        """
        files = [file for file in os.listdir(self.checkpoint_dir) if (file.endswith(".pt") or file.endswith(".tar"))]
        filepaths = [os.path.join(self.checkpoint_dir, file) for file in files]
        last_file = max(filepaths, key=os.path.getctime)
        return os.path.abspath(last_file)

    def load_checkpoint(self, checkpoint_path=None):
        """
        Saving the networks and all parameters from a given path.
        If the given path is None, then the latest saved file in 'checkpoint_dir' will be used.
        :param checkpoint_path: File to load the model from
        :return:
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_path_of_latest_file()

        if os.path.isfile(checkpoint_path):
            logger.info('Loading checkpoint...({})'.format(checkpoint_path))
            key = 'cuda' if torch.cuda.is_available() else 'cpu'

            checkpoint = torch.load(checkpoint_path, map_location=key)
            start_timestep = checkpoint['last_timestep'] + 1
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            replay_buffer = checkpoint['replay_buffer']

            gc.collect()
            logger.info('Loaded model at timestep {} from {}'.format(start_timestep, checkpoint_path))
            return start_timestep, replay_buffer
        else:
            raise OSError('Checkpoint not found')

    def set_eval(self):
        """
        Sets the model in evaluation mode
        :return:
        """
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def set_train(self):
        """
        Sets the model in training mode
        :return:
        """
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()

    def get_network(self, name):
        if name == 'Actor':
            return self.actor
        elif name == 'Critic':
            return self.critic
        else:
            raise NameError('name \'{}\' is not defined as a network'.format(name))