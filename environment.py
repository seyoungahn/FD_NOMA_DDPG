# -*- coding: utf-8 -*-
# Reinforcement learning libraries
import gym
import numpy as np
from gym import Env
from gym import spaces
import csv
import time

# System models
import utils
from cell import Cell

# Utilities
import csv
import copy
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import init

# System configurations
import json
with open('params.json', 'r') as f:
    params = json.load(f)

class CellularNetworksEnvironment(Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(CellularNetworksEnvironment, self).__init__()
        self.cell = Cell()
        self.done = False
        self.info = {}
        self.experience_count = 0
        self.writer = SummaryWriter('runs/run_system')
        # self.record_flag = False
        # Observation: Cell 객체에서 수집
        # [RX1_CSI, RX2_CSI, RX3_CSI, RX4_CSI,
        #  RX1_SE, RX2_SE, RX3_SE, RX4_SE,
        #  V2V_dist, V2I_dist, DLG_dist, CCI_dist]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([2.5e+11, 2.5e+11, 2.5e+11, 2.5e+11, 8.0, 8.0, 8.0, 8.0, 500.0 * np.sqrt(2.0), 500.0 * np.sqrt(2.0), 500.0 * np.sqrt(2.0), 500.0 * np.sqrt(2.0)]),
            dtype=np.float64
        )
        # Action: 4-elements (continuous values)
        # [secondary_power, V2V_power, V2I_power, BS_power]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float64
        )
        # print(self.observation_space.sample())

    def reset(self):
        self.done = False
        self.experience_count = 0
        self.cell.refresh_cell()
        self.cell.set_cell_state()
        csi = self.cell.get_CSI()
        efficiency = self.cell.get_spectral_efficiency(csi)
        distance = self.cell.get_dist()
        observation = np.array(csi + efficiency + distance)
        # print(observation)
        # print(self.observation_space.contains(observation))
        # reward, done, info cannot be included
        return observation

    def step(self, action):
        if self.done:
            print("EPISODE DONE.")
        elif self.experience_count == params['EPI']:
            self.done = True
        else:
            self.experience_count += 1
        self.cell.actions = action
        self.cell.set_cell_state()
        csi = self.cell.get_CSI()
        efficiency = self.cell.get_spectral_efficiency(csi)
        distance = self.cell.get_dist()
        state = np.array(csi + efficiency + distance)
        reward, status = self.get_reward(state, action)
        if self.done:
            ### Performance plotting
            # status = [tx_power, efficiency, qos, ACK, reward], 각 element는 RX1 ~ RX4
            self.writer.add_scalar('epoch/Spectral efficiency (RX1)', status[1][0], init.EPOCHES)
            self.writer.add_scalar('epoch/Spectral efficiency (RX2)', status[1][1], init.EPOCHES)
            self.writer.add_scalar('epoch/Spectral efficiency (RX3)', status[1][2], init.EPOCHES)
            self.writer.add_scalar('epoch/Spectral efficiency (RX4)', status[1][3], init.EPOCHES)
            self.writer.add_scalar('epoch/Spectral efficiency (V2V)', status[1][0] + status[1][3], init.EPOCHES)
            self.writer.add_scalar('epoch/Spectral efficiency (V2I)', status[1][1] + status[1][2], init.EPOCHES)
            self.writer.add_scalar('epoch/Transmit power (secondary)', status[0][0], init.EPOCHES)
            self.writer.add_scalar('epoch/Transmit power (V2V)', status[0][1], init.EPOCHES)
            self.writer.add_scalar('epoch/Transmit power (V2I)', status[0][2], init.EPOCHES)
            self.writer.add_scalar('epoch/Transmit power (BS)', status[0][3], init.EPOCHES)
            # print(status)
            with open('./log/system_status.csv', 'a+', encoding='utf-8', newline='') as ptr:
                wr = csv.writer(ptr)
                elem = []
                elem.extend([init.EPOCHES])
                elem.extend(status[0])
                elem.extend(status[1])
                elem.extend(status[2])
                elem.extend(status[3])
                # print(elem)
                wr.writerow(elem)
            init.EPOCHES += 1
        self.cell.move_vues()
        return state, reward, self.done, self.info

    def get_reward(self, state, action):
        tx_power = utils.action2power(action)
        # tx_power = action
        dict_power = {}
        for i in range(4):
            dict_power[i] = tx_power[i]
        dict_power_idx = np.array(sorted(dict_power.items(), key=lambda item: item[1])).transpose()[0]
        ACK = self._get_ACK(dict_power_idx)
        # SIC constraint
        ACK = self._sic_constraint(ACK, state, action)
        efficiency = state[4:8]
        qos = [params['v2v_ul_qos'], params['v2i_ul_qos'], params['v2i_dl_qos'], params['primary_qos']]
        reward = []
        for i in range(4):
            reward.append((efficiency[i] / qos[i]) * ACK[i])
        reward = np.array(reward)
        # if self.record_flag:
        #     # filename = 'log/result.csv'.format(time.strftime('%a%d%b%Y%H%M%SGMT', time.localtime()))
        #     with open('./log/result.csv', 'a+', encoding='utf-8', newline='') as f:
        #         rdr = csv.writer(f)
        #         data = []
        #         for elem in tx_power:
        #             data.append(elem)
        #         for elem in efficiency:
        #             data.append(elem)
        #         for elem in qos:
        #             data.append(elem)
        #         for elem in ACK:
        #             data.append(elem)
        #         for elem in reward:
        #             data.append(elem)
        #         rdr.writerow(data)
        #     self.record_flag = False
        reward = reward.mean()
        status = [tx_power, efficiency, qos, ACK, reward]
        return reward, status

    def _get_ACK(self, idx):
        ptr = -1
        temp = [0, 0, 0, 0]
        for i in range(4):
            if np.where(idx == i)[0] < ptr:
                continue
            ptr = np.where(idx == i)[0]
            temp[i] = 1
        result = [temp[1], temp[2], temp[3], temp[0]]
        # print(result)
        return result

    def _sic_constraint(self, ack, state, action):
        ACK = ack
        tx_power = utils.action2power(action)
        dist = state[8:]
        dist_secondary = 0.0
        primary_rx_power = 0.0
        if len(self.cell.secondary_vue_idx) != 0:
            for i in self.cell.secondary_vue_idx:
                dist_secondary += utils.Euclidean_dist(self.cell.vue_list[self.cell.primary_vue_idx].get_abs_pos(), self.cell.vue_list[i].get_abs_pos())
            dist_secondary /= len(self.cell.secondary_vue_idx)
            primary_rx_power = tx_power[0] - utils.pathloss(dist_secondary, 'V2V', 'LOS')
        V2V_rx_power = tx_power[1] - utils.pathloss(dist[0], 'V2V', 'LOS')
        V2I_rx_power = tx_power[2] - utils.pathloss(dist[1], 'V2I', 'LOS')
        if abs(V2I_rx_power - V2V_rx_power) < params['sic_tolerance']:
            ACK = [0, 0, 0, 0]
        if abs(V2V_rx_power - primary_rx_power) < params['sic_tolerance']:
            ACK = [0, 0, 0, 0]

        # if self.record_flag:
        #     # filename = 'log/result.csv'.format(time.strftime('%a%d%b%Y%H%M%SGMT', time.localtime()))
        #     with open('./log/result_rx.csv', 'a+', encoding='utf-8', newline='') as f:
        #         rdr = csv.writer(f)
        #         data = [primary_rx_power, V2V_rx_power, V2I_rx_power]
        #         rdr.writerow(data)
        return ACK

    def render(self, mode='human'):
        self.cell.get_status()

    def close(self):

        return 0

## Check code
# from stable_baselines3.common.env_checker import check_env
# env = CellularNetworksEnvironment()
# check_env(env)