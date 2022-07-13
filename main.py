# -*- coding: utf-8 -*-
import csv
import logging

import numpy as np
import os
import random
import time

import torch
from matplotlib import pyplot as plt
import datetime as dt

from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import gym
from model import DDPG
from utils import OrnsteinUhlenbeckActionNoise
from replay_memory import ReplayMemory, Transition
from utils import NormalizedActions

from cell import Cell
from environment import CellularNetworksEnvironment

# System configurations
import json
with open('params.json', 'r') as f:
    params = json.load(f)
SEED_VALUE = 0
load_model = False
render_train = False
render_eval = False

# Logger settings
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# GPU settings
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info('USING {}'.format(dev))

# Environment registration
from gym.envs.registration import register
register(
    id='cellularnetworks-v0',
    entry_point='environment:CellularNetworksEnvironment'
)

if __name__ == '__main__':
    # Define the directory where to save and load models
    checkpoint_dir = params['savedir'] + params['env']
    writer = SummaryWriter('runs/run_test1')

    # Create the env
    # env = CellularNetworksEnvironment()
    env = gym.make('cellularnetworks-v0')
    env = NormalizedActions(env)

    # Define the reward threshold when the task is solved (if existing) for model saving
    reward_threshold = np.inf

    # Set random seed for all used libraries where possible
    # env.seed(SEED_VALUE)
    # torch.manual_seed(SEED_VALUE)
    # np.random.seed(SEED_VALUE)
    # random.seed(SEED_VALUE)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(SEED_VALUE)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    # Define and build DDPG agent
    hidden_size = tuple(params['HIDDEN_SIZE'])
    agent = DDPG(params['GAMMA'], params['TAU'], hidden_size, env.observation_space.shape[0], env.action_space, checkpoint_dir=checkpoint_dir)

    # Initialize replay memory
    memory = ReplayMemory(int(params['REPLAY_SIZE']))

    nb_actions = env.action_space.shape[-1]
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(params['NOISE_STDDEV']) * np.ones(nb_actions))

    # Define counters and other variables
    start_step = 0
    if load_model:
        start_step, memory = agent.load_checkpoint()
    timestep = start_step // 10000 + 1
    rewards, policy_losses, value_losses, mean_test_rewards = [], [], [], []
    epoch = 0
    t = 0
    time_last_checkpoint = time.time()

    # Start training
    # logger.info('TRAIN AGENT ON {} ENV'.format({env.unwrapped.spec.id}))
    logger.info('DOING {} TIMESTEPS'.format(params['TIMESTEPS']))
    logger.info("START AT TIMESTEP {0} WITH t = {1}".format(timestep, t))
    logger.info("START TRAINING AT {}".format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

    while timestep <= params["TIMESTEPS"]:
        ou_noise.reset()
        epoch_return = 0.0
        epoch_value_loss = 0.0
        epoch_policy_loss = 0.0

        state = torch.Tensor(np.array([env.reset()])).to(dev)
        # print("TIMESTEP: {}".format(timestep))
        cnt = 1
        while True:
            if render_train:
                env.render()

            # if cnt == params['EPI']:
            #     env.record_flag = True
            #     cnt = 1
            # else:
            #     cnt = cnt + 1

            action = agent.calc_action(state, ou_noise)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            timestep += 1
            epoch_return += reward

            # print(done)

            mask = torch.Tensor(np.array([done])).to(dev)
            reward = torch.Tensor(np.array([reward])).to(dev)
            next_state = torch.Tensor(np.array([next_state])).to(dev)

            memory.push(state, action, mask, next_state, reward)

            state = next_state

            if len(memory) > params['BATCH_SIZE']:
                transitions = memory.sample(params['BATCH_SIZE'])
                batch = Transition(*zip(*transitions))

                value_loss, policy_loss = agent.update_params(batch)

                epoch_value_loss += value_loss
                epoch_policy_loss += policy_loss

            if done:
                break

        epoch_return /= params['EPI']
        epoch_value_loss /= params['EPI']
        epoch_policy_loss /= params['EPI']

        # env.render()
        rewards.append(epoch_return)
        value_losses.append(epoch_value_loss)
        policy_losses.append(epoch_policy_loss)
        writer.add_scalar('epoch/mean reward', epoch_return, epoch)
        writer.add_scalar('epoch/mean value loss', epoch_value_loss, epoch)
        writer.add_scalar('epoch/mean policy loss', epoch_policy_loss, epoch)
        with open('./log/learning_status.csv', 'a+', encoding='utf-8', newline='') as ptr:
            wr = csv.writer(ptr)
            wr.writerow([epoch, epoch_return, epoch_value_loss, epoch_policy_loss])

        # Test every 10-th episode (== 10,000) steps for a number of test_epochs epochs
        if timestep >= 10000 * t:
            t += 1
            test_rewards = []
            for _ in range(params["N_TEST_CYCLES"]):
                state = torch.Tensor(np.array([env.reset()])).to(dev)
                test_reward = 0
                while True:
                    if render_eval:
                        env.render()

                    action = agent.calc_action(state) # Selection without noise

                    next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
                    test_reward += reward

                    next_state = torch.Tensor(np.array([next_state])).to(dev)

                    state = next_state
                    if done:
                        break
                test_rewards.append(test_reward)

            mean_test_rewards.append(np.mean(test_rewards))

            for name, param in agent.actor.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            for name, param in agent.critic.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

            writer.add_scalar('test/mean_test_return', mean_test_rewards[-1], epoch)
            logger.info('EPOCH: {}, CURRENT TIMESTEP: {}, LAST REWARD: {}, MEAN REWARD: {}, MEAN TEST REWARD: {}'.format(epoch, timestep, rewards[-1], np.mean(rewards[-10:]), np.mean(test_rewards)))

            # Save if the mean of the last three averaged rewards while testing is greater than the specified reward threshold
            if np.mean(mean_test_rewards[-3:]) >= reward_threshold:
                agent.save_checkpoint(timestep, memory)
                time_last_checkpoint = time.time()
                logger.info('SAVED MODEL AT {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

        epoch += 1

    agent.save_checkpoint(timestep, memory)
    logger.info('SAVED MODEL AT ENDTIME {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
    logger.info('STOPPING TRAINING AT {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
    env.close()
