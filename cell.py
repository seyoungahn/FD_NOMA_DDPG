# -*- coding: utf-8 -*-
import random
import numpy as np
from matplotlib import pyplot as plt

import init
import utils
import json
import vue

with open('params.json', 'r') as f:
    params = json.load(f)

class Cell:
    def __init__(self):
        # Constant values
        self.BS_height = params["BS_height"]
        self.n_vue = params['n_vue']  ## n_vue = target_vue + vue_list 길이
        self.BS_coordi = [0.0, 0.0]
        self.SI_coefficient_BS = params['si_coefficient_bs']
        self.SI_coefficient_VUE = params['si_coefficient_vue']
        # Variables
        self.vue_list = [vue.VUE(i) for i in range(self.n_vue)]
        self.primary_vue_idx = 0
        self.secondary_vue_idx = set()
        self.dl_user_idx = []
        self.v2v_vue_idx = 0
        self.actions = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]

    def set_downlink_users(self):
        # target_vue가 속한 zone에 대한 downlink user group의 zone number들
        if 0 < self.vue_list[0].zone_num <= 97:
            opposite_zone_num = self.vue_list[0].zone_num + 98
        elif 98 < self.vue_list[0].zone_num <= 195:
            opposite_zone_num = self.vue_list[0].zone_num - 98
        else:
            assert "ERROR: zone_num is out of bound."
        opposite_zone_bp = get_zone_bp(opposite_zone_num)
        opposite_polar = utils.cart2pol(opposite_zone_bp)
        if np.pi/16 < opposite_polar[1] < 2 * np.pi - np.pi/16:
            angle_range = [opposite_polar[1] - np.pi/16, opposite_polar[1] + np.pi/16]
            for vue in self.vue_list:
                p = vue.get_abs_pos()
                p = utils.cart2pol(p)
                if angle_range[0] < p[1] < angle_range[1]:
                    self.dl_user_idx.append(vue.id)
        else:
            angle_range = [opposite_polar[1] - np.pi / 16, opposite_polar[1] + np.pi / 16]
            for vue in self.vue_list:
                p = vue.get_abs_pos()
                p = utils.cart2pol2(p)
                if angle_range[0] < p[1] < angle_range[1]:
                    self.dl_user_idx.append(vue.id)


    def get_expected_dl_dist(self):
        dl_zone_set = set()
        dl_user_num = {}
        for i in self.dl_user_idx:
            dl_zone_set.add(self.vue_list[i].zone_num)
            if self.vue_list[i].zone_num in dl_user_num:
                dl_user_num[self.vue_list[i].zone_num] += 1
            else:
                dl_user_num[self.vue_list[i].zone_num] = 1
        expected_dist = 0
        if len(self.dl_user_idx) != 0:
            for dl_zone_num in dl_zone_set:
                distance = utils.Euclidean_dist(self.vue_list[self.primary_vue_idx].zone_bp, get_zone_bp(dl_zone_num))
                expected_dist += dl_user_num[dl_zone_num] * distance
            expected_dist = expected_dist / len(self.dl_user_idx)
        return expected_dist

    def get_status(self):
        # point_list = []
        # for car in self.vue_list:
        #     abs_coordi = car.get_abs_pos()
        #     point_list.append(abs_coordi)
        #     print("VUE #{} => zone_num: {}, zone_bp: ({}, {}), zone_coordi: ({}, {}) => absolute position: ({}, {})".format(car.id, car.zone_num, car.zone_bp[0], car.zone_bp[1], car.zone_coordi[0], car.zone_coordi[1], abs_coordi[0], abs_coordi[1]))
        # point_list = np.transpose(point_list)
        self.draw_system()
        plt.show()

    def get_CSI(self):
        #### State 정보 가운데 CSI 정보인 SINR 계산 (dBm scale)
        tx_power = utils.action2power(self.actions)
        # tx_power = self.actions
        noise = utils.dBm2mW(params['white_noise']) * params["BW"] * 10e+6

        #### RX1: V2V uplink receiver VUE
        ## Interference source: downlink signal
        dist2D_V2V = utils.Euclidean_dist(self.vue_list[self.primary_vue_idx].get_abs_pos(), self.vue_list[self.v2v_vue_idx].get_abs_pos())
        RX1_channel_gain = utils.pathloss(dist2D_V2V, 'V2V', 'LOS')
        # dist2D_CCI = utils.Euclidean_dist(self.BS_coordi, self.vue_list[self.v2v_vue_idx].get_abs_pos())
        # CCI_interference = tx_power[3] - utils.pathloss(dist2D_CCI, 'V2I', 'NLOS')  # BS의 downlink signal로부터 받는 interference
        center_pos = get_zone_bp(self.vue_list[self.primary_vue_idx].zone_num) # primary VUE가 속한 zone의 중점 좌표
        for i in range(2):
            center_pos[i] += 5.0
        dist2D_SIC = utils.Euclidean_dist(self.vue_list[self.v2v_vue_idx].get_abs_pos(), center_pos)
        NOMA_noise = tx_power[0] - utils.pathloss(dist2D_SIC, 'V2V', 'LOS')
        # RX1_interference = CCI_interference + SIC_interference
        RX1_interference = NOMA_noise
        RX1_CSI = np.divide(utils.dBm2mW(tx_power[1] - RX1_channel_gain), (utils.dBm2mW(RX1_interference) + noise))

        #### RX2: V2I uplink receiver BS
        ## Interference source: downlink signal (self-interference)
        dist2D_V2I = utils.Euclidean_dist(self.BS_coordi, self.vue_list[self.primary_vue_idx].get_abs_pos())
        RX2_channel_gain = utils.pathloss(dist2D_V2I, 'V2I', 'LOS')
        NOMA_noise = utils.mW2dBm(utils.dBm2mW(tx_power[0]) + utils.dBm2mW(tx_power[1])) - utils.pathloss(dist2D_V2I, 'V2I', 'LOS')
        RX2_interference = utils.dBm2mW(utils.self_interference_cancellation(utils.dBm2mW(tx_power[3]))) + utils.dBm2mW(NOMA_noise) # self-interference + NOMA SIC interference
        RX2_CSI = np.divide(utils.dBm2mW(tx_power[2] - RX2_channel_gain), (RX2_interference + noise))

        #### RX3: V2I downlink receivers downlink user group
        ## Interference source: uplink signal (co-channel interference), downlink signal (other signals)
        if len(self.dl_user_idx) != 0:
            expected_dl_dist2D = self.get_expected_dl_dist()
            if expected_dl_dist2D != 0:
                dl_channel_gain = utils.pathloss(expected_dl_dist2D, 'V2V', 'NLOS')
            else:
                dl_channel_gain = 0
            primary_interference = utils.mW2dBm(utils.dBm2mW(tx_power[1]) + utils.dBm2mW(tx_power[2])) - dl_channel_gain
            secondary_interference = tx_power[0] - dl_channel_gain
            RX3_interference = utils.dBm2mW(primary_interference) + utils.dBm2mW(secondary_interference)
            RX3_channel_gain = 0.0
            for i in self.dl_user_idx:
                dist2D_DLG = utils.Euclidean_dist(self.BS_coordi, self.vue_list[i].get_abs_pos())
                RX3_channel_gain += utils.dBm2mW(utils.pathloss(dist2D_DLG, 'V2I', 'LOS'))
            RX3_channel_gain = utils.mW2dBm(np.divide(RX3_channel_gain, len(self.dl_user_idx)))
            RX3_CSI = np.divide(utils.dBm2mW(tx_power[3] - RX3_channel_gain), RX3_interference + noise)
        else:
            RX3_CSI = 0.0

        #### RX4: V2V uplink receiver primary VUE
        ## Interference source: uplink signal (co-channel interference)
        RX4_CSI = 0.0
        # dist2D_CCI = utils.Euclidean_dist(self.BS_coordi, self.vue_list[self.primary_vue_idx].get_abs_pos())
        # RX4_CCI = tx_power[3] - utils.pathloss(dist2D_CCI, 'V2I', 'NLOS')
        if len(self.secondary_vue_idx) != 0:
            for i in self.secondary_vue_idx:
                dist2D_secondary = utils.Euclidean_dist(self.vue_list[self.primary_vue_idx].get_abs_pos(), self.vue_list[i].get_abs_pos())
                RX4_channel_gain = utils.pathloss(dist2D_secondary, 'V2V', 'LOS')
                RX4_SI = utils.self_interference_cancellation(utils.dBm2mW(tx_power[1]) + utils.dBm2mW(tx_power[2]))
                # RX4_CSI += utils.dBm2mW(tx_power[0] - channel_gain) / (utils.dBm2mW(RX4_SI) + utils.dBm2mW(RX4_CCI) + noise)
                # print("SINR: {}".format(utils.dBm2mW(tx_power[0] - RX4_channel_gain) / (utils.dBm2mW(RX4_SI) + noise)))
                # print("tx_power: {}".format(tx_power[0]))
                # print("channel gain: {}".format(RX4_channel_gain))
                # print("self-interference: {}".format(RX4_SI))
                # print("dist: {}".format(dist2D_secondary))
                RX4_CSI += np.divide(utils.dBm2mW(tx_power[0] - RX4_channel_gain), utils.dBm2mW(RX4_SI) + noise)
            RX4_CSI = np.divide(RX4_CSI, len(self.secondary_vue_idx))

        ## Interference source 2: downlink NOMA group으로 부터 받는 co-channel interference
        ## downlink user들 가운데 BS와 가장 가깝게 있는
        ## self.actions = [V2I uplink power, V2V uplink power, secondary power]
        return [RX1_CSI, RX2_CSI, RX3_CSI, RX4_CSI]

    def get_spectral_efficiency(self, CSI):
        #### RX1: V2V uplink receiver VUE
        RX1_SE = np.log2(1 + CSI[0])
        #### RX2: V2I uplink receiver BS
        RX2_SE = np.log2(1 + CSI[1])
        #### RX3: V2I downlink receivers downlink user group
        RX3_SE = np.log2(1 + CSI[2])
        #### RX4: V2V uplink receiver primary VUE (sum capacity)
        RX4_SE = np.log2(1 + CSI[3])
        return [RX1_SE, RX2_SE, RX3_SE, RX4_SE]

    def get_dist(self):
        primary_point = self.vue_list[self.primary_vue_idx].get_abs_pos()
        v2v_point = self.vue_list[self.v2v_vue_idx].get_abs_pos()
        secondary_points = []
        for i in self.secondary_vue_idx:
            secondary_points.append(self.vue_list[i].get_abs_pos())
        #### V2V: V2V user와 primary user 사이 거리
        V2V_dist = utils.Euclidean_dist(v2v_point, primary_point)
        #### V2I: BS와 primary user 사이 거리
        V2I_dist = utils.Euclidean_dist(self.BS_coordi, primary_point)
        #### DLG: BS와 Downlink user group 사이 평균 거리 (BS는 DLG user들의 정보를 모두 알고있음)
        DLG_dist = 0.0
        if len(self.dl_user_idx) != 0:
            for i in self.dl_user_idx:
                DLG_dist += utils.Euclidean_dist(self.BS_coordi, self.vue_list[i].get_abs_pos())
            DLG_dist = DLG_dist / len(self.dl_user_idx)
        #### CCI: DLG와 ULG 사이의 평균 거리 (primary VUE는 DLG user들의 정보를 모름)
        CCI_dist = self.get_expected_dl_dist()
        return [V2V_dist, V2I_dist, DLG_dist, CCI_dist]

    def set_cell_state(self):
        #### Cell 내 정보 설정 ####
        ## 0: Cell parameter 초기화
        self.primary_vue_idx = 0
        self.secondary_vue_idx = set()
        self.dl_user_idx = []
        self.v2v_vue_idx = 0

        ## 1: Target VUE가 속한 zone 내 primary VUE, secondary VUE 선정
        zone_user_idx = set()
        for i in range(params['n_vue']):
            if self.vue_list[i].zone_num == self.vue_list[0].zone_num:
                zone_user_idx.add(self.vue_list[i].id)

        if 0 < self.vue_list[0].zone_num <= 48:
            max_dist = 0.0
            for i in zone_user_idx:
                if self.vue_list[i].zone_coordi[1] > max_dist:
                    max_dist = self.vue_list[i].zone_coordi[1]
                    self.primary_vue_idx = self.vue_list[i].id
        elif 49 < self.vue_list[0].zone_num <= 97:
            max_dist = 10.0
            for i in zone_user_idx:
                if self.vue_list[i].zone_coordi[0] < max_dist:
                    max_dist = self.vue_list[i].zone_coordi[0]
                    self.primary_vue_idx = self.vue_list[i].id
        elif 98 < self.vue_list[0].zone_num <= 146:
            max_dist = 10.0
            for i in zone_user_idx:
                if self.vue_list[i].zone_coordi[1] < max_dist:
                    max_dist = self.vue_list[i].zone_coordi[1]
                    self.primary_vue_idx = self.vue_list[i].id
        elif 147 < self.vue_list[0].zone_num <= 195:
            max_dist = 0.0
            for i in zone_user_idx:
                if self.vue_list[i].zone_coordi[0] > max_dist:
                    max_dist = self.vue_list[i].zone_coordi[0]
                    self.primary_vue_idx = self.vue_list[i].id
        else:
            assert "ERROR: zone_num is zero."

        self.secondary_vue_idx = zone_user_idx - {self.primary_vue_idx}

        ## 2: Target VUE가 속한 zone에 대한 downlink user group 설정
        self.set_downlink_users()

        ## 3: V2V rx VUE 설정
        self.v2v_vue_idx = self.get_nearest_vues(self.primary_vue_idx)

    def add_vue_list(self, vue):
        self.vue_list.append(vue)

    def get_nearest_vues(self, tx_vue_id):
        vues_by_zone = [[] for _ in range(params['n_zone'])]
        for car in self.vue_list:
            vues_by_zone[car.zone_num].append(car.id)
        nearest_vue_id = None
        min_dist = 100000
        zone_ptr = self.vue_list[tx_vue_id].zone_num
        flag = True
        while flag:
            if zone_ptr == 1:
                zone_ptr = 195
            else:
                zone_ptr -= 1
            for i in vues_by_zone[zone_ptr]:
                p1 = self.vue_list[tx_vue_id].get_abs_pos()
                p2 = self.vue_list[i].get_abs_pos()
                dist = utils.Euclidean_dist(p1, p2)
                if dist < min_dist:
                    nearest_vue_id = i
                flag = False
        return nearest_vue_id

    def move_vues(self):
        for vue in self.vue_list:
            lane = random.uniform(0.0, 10.0)
            speed = random.uniform(0.0, 100.0)
            dist = utils.forward_dist(speed, params['config_period'])
            next_zone = vue.zone_num - int(abs(dist / params['zone_scale']))

            if next_zone < 0:
                next_zone = 195 + next_zone
            next_coordi = dist % params['zone_scale']

            if next_zone % 49 == 0:
                next_zone += 1
            vue.zone_num = next_zone

            if 0 < vue.zone_num <= 48:
                vue.zone_coordi = [lane, next_coordi]
            elif 49 < vue.zone_num <= 97:
                vue.zone_coordi = [params['zone_scale'] - next_coordi, lane]
            elif 98 < vue.zone_num <= 146:
                vue.zone_coordi = [lane, params['zone_scale'] - next_coordi]
            elif 147 < vue.zone_num <= 195:
                vue.zone_coordi = [next_coordi, lane]
            else:
                assert "ERROR: Invalid zone_num"
            vue.zone_bp = get_zone_bp(vue.zone_num)

    def refresh_cell(self):
        # VUE 위치 재배치
        init.ZONE_QUEUE = [0 for _ in range(params['n_zone'])]
        for vue in self.vue_list:
            vue.set_zone_num()
            vue.set_random_pos()

    def test_cell(self):
        ## 확인해봐야 할 부분
        return 0

    def draw_system(self):
        plt.figure(figsize=(15, 15))

        def connect_points(x, y, p1, p2, color, linestyle):
            x1, x2 = x[p1], x[p2]
            y1, y2 = y[p1], y[p2]
            plt.plot([x1, x2], [y1, y2], color=color, linestyle=linestyle, linewidth=0.5, zorder=1)

        outline_x = [-250, -250, 250, 250, -250, 250, -250, 250]
        outline_y = [250, -250, -250, 250, 250, 250, -250, -250]
        inline_x = [-240, -240, 240, 240, 240, -240, 240, -240]
        inline_y = [240, -240, 240, -240, 240, 240, -240, -240]

        for i in np.arange(0, len(outline_x), 2):
            connect_points(outline_x, outline_y, i, i + 1, 'black', '-')
        for i in np.arange(0, len(inline_x), 2):
            connect_points(inline_x, inline_y, i, i + 1, 'black', '-')

        for car in self.vue_list:
            point = car.get_abs_pos()
            # VUE plotting
            if car.id == 0:
                plt.plot(point[0], point[1], color='red', marker='o', linestyle='None', zorder=3, markersize=6,
                         markeredgewidth=0.2, markeredgecolor='white')
            elif car.id in self.dl_user_idx:
                plt.plot(point[0], point[1], color='green', marker='o', linestyle='None', zorder=3, markersize=6,
                         markeredgewidth=0.2, markeredgecolor='white')
            else:
                plt.plot(point[0], point[1], color='blue', marker='o', linestyle='None', zorder=3, markersize=6,
                         markeredgewidth=0.2, markeredgecolor='white')
            # BS plotting
            plt.plot(0, 0, color='black', marker='^', linestyle='None', zorder=3, markersize=10, markeredgewidth=1,
                     markeredgecolor='white')

def get_zone_bp(zone_num):
    if 0 < zone_num <= 48:
        return [-250.0, 240.0 - 10.0 * zone_num]
    elif 49 < zone_num <= 97:
        return [-740.0 + 10.0 * zone_num, -250.0]
    elif 98 < zone_num <= 146:
        return [240.0, -1230.0 + 10.0 * zone_num]
    elif 147 < zone_num <= 195:
        return [1710.0 - 10.0 * zone_num, 240.0]
    else:
        assert "ERROR: zone_num is zero."