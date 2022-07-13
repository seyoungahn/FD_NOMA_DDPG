# -*- coding: utf-8 -*-

from math import *

import gym
import numpy as np
import random
from matplotlib import pyplot as plt

# System configurations
import json
with open('params.json', 'r') as f:
    params = json.load(f)

def cart2pol(point):
    rho = np.sqrt(point[0]**2 + point[1]**2)
    phi = np.arctan2(point[1], point[0])
    if phi < 0:
        phi = 2 * np.pi + phi
    return [rho, phi]

def cart2pol2(point):
    rho = np.sqrt(point[0]**2 + point[1]**2)
    phi = np.arctan2(point[1], point[0])
    return [rho, phi]

def pol2cart(point):
    x = point[0] * np.cos(point[1])
    y = point[0] * np.sin(point[1])
    return [x, y]

def mW2dBm(mW):
    return 10.0 * np.log10(mW)

def dBm2mW(dBm):
    return 10.0 ** (dBm / 10)

def forward_dist(kmh, ms):
    # km/h => m/ms * ms = meter
    return kmh * (1.0/3600.0) * ms

def Euclidean_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_lognormal_value(mu, sigma):
    normal_std = np.sqrt(np.log10(1 + (sigma/mu)**2))
    normal_mean = np.log10(mu) - normal_std**2 / 2
    return np.random.lognormal(normal_mean, normal_std)

def pathloss(dist2D, mode, state):
    # Assuming only urban case
    assert dist2D != 0, "ERROR: Distance is zero"
    dist3D = np.sqrt((dist2D)**2 + (params["BS_height"] - params["RF_height"])**2)
    if mode == "V2V":
        if state == 'LOS':
            return 38.77 + 16.7 * np.log10(dist3D) + 18.2 * np.log10(params['center_freq_V2V']) + np.random.normal(0.0, 3.0)
            # return 38.77 + 16.7 * np.log10(dist3D) + 18.2 * np.log10(params['center_freq_V2V']) + get_lognormal_value(0.0, 3.0)
            # return 40.0 * np.log10(dist3D) + 9.45 - 17.3 * np.log10(params['BS_height']) - 17.3 * np.log10(params['RF_height']) + 2.7 * np.log10(params['center_freq_V2V'] * 1e+9 / 5.0) + np.random.normal(0.0, 3.0)
        else:
            return 36.85 + 30.0 * np.log10(dist3D) + 18.9 * np.log10(params['center_freq_V2V']) + np.random.normal(0.0, 4.0)
            # return 36.85 + 30.0 * np.log10(dist3D) + 18.9 * np.log10(params['center_freq_V2V']) + get_lognormal_value(0.0, 4.0)
    elif mode == "V2I":
        bp = 4 * (params["BS_height"] - 1) * (params["RF_height"] - 1) * (params["center_freq_V2I"] * 1.0e9 / params["c"])
        if dist2D <= bp:
            # PL1
            UMa_LOS = 28.0 + 22.0 * np.log10(dist3D) + 20.0 * np.log10(params["center_freq_V2I"])
        else:
            # PL2
            UMa_LOS = 28.0 + 40.0 * np.log10(dist3D) + 20.0 * np.log10(params["center_freq_V2I"]) - 9.0 * np.log10((bp)**2 + (params["BS_height"] - params["RF_height"])**2)
        if state == 'LOS':
            return UMa_LOS + np.random.normal(0.0, 4.0)
            # return UMa_LOS + get_lognormal_value(0.0, 4.0)
            # return 128.1 + 37.6 * np.log10(dist3D) + np.random.normal(0.0, 4.0)
        else:
            UMa_NLOS = 13.54 + 39.08 * np.log10(dist3D) + 20.0 * np.log10(params["center_freq_V2I"]) - 0.6 * (params["RF_height"] - 1.5)
            return max(UMa_LOS, UMa_NLOS) + np.random.normal(0.0, 6.0)
            # return max(UMa_LOS, UMa_NLOS) + get_lognormal_value(0.0, 6.0)
    else:
        assert True, "ERROR: invalid path loss mode"

def action2power(actions):
    # VUE transmit power: -1.0 <= action <= 1.0 --- 0.0 <= power <= 23.0
    # BS transmit power: -1.0 <= action <= 1.0 --- 0.0 <= power <= 43.0
    power = []
    for i in range(3):
        power.append((actions[i] + 1.0) * (params['max_power'] / 2.0))
    power.append((actions[3] + 1.0) * (params['BS_tx_power'] / 2.0))
    return np.array(power)

def self_interference_cancellation(self_interference):
    self_interference = mW2dBm(self_interference)
    return self_interference - params['si_tolerance']

##### Training utilities
class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        """
        Normalizes the actions to be in between action_space.high and action_space.low.
        If action_space.low == -action_space.high, this is equals to action_space.high * action.
        :param action:
        :return: normalized action
        """
        action = (action + 1) / 2 # [-1, 1] => [0, 1]
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def reverse_action(self, action):
        """
        Reverts the normalization
        :param action:
        :return:
        """
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action

# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
# and adapted to be synchronous with https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OUNoise:
    def __init__(self, action_dimension, dt=0.01, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.dt = dt
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.random.randn(len(x)) * np.sqrt(self.dt)
        self.state = x + dx
        return self.state


# From OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def noise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# From OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.2, adaptation_coefficient=1.01):
        """
        Note that initial_stddev and current_stddev refer to std of parameter noise,
        but desired_action_stddev refers to (as name notes) desired std in action space
        """
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adaptation_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adaptation_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adaptation_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adaptation_coefficient)


def ddpg_distance_metric(actions1, actions2):
    """
    Compute "distance" between actions taken by two policies at the same states
    Expects numpy arrays
    """
    diff = actions1 - actions2
    mean_diff = np.mean(np.square(diff), axis=0)
    dist = sqrt(np.mean(mean_diff))
    return dist