# -*- coding: utf-8 -*-

"""
Assumption: 1 pixel = 1 meter
N: number of cars
INTERVAL: time interval of a single network snapshot
SPEED: number of pixels the VUE moves per one interval
The wider INTERVAL, the slower SPEED (SPEED * 100 = INTERVAL: 36km/h)
<< settings >>
(INTERVAL = 100, SPEED = 1) or (INTERVAL = 200, SPEED = 2)
"""

# First set up the figure, axis, and plot element we want to animate
import random
import time

from init import *

# System configurations
import json
with open('params.json', 'r') as f:
    params = json.load(f)

# 0: 직진(50%), 1: 좌회전(25%), 2: 우회전(25%)
def choose_direction():
    randnum = random.randrange(1, 100)
    if randnum <= 50:
        return 0
    elif 50 < randnum <= 75:
        return 1
    elif 75 < randnum:
        return 2
    else:
        pass

class VUE:
    def __init__(self, id):
        self.id = id
        self.prev_zone_num = None
        self.prev_zone_coordi = None
        self.zone_num = None
        self.zone_coordi = None
        self.set_zone_num()
        self.set_random_pos()
        self.is_primary = False
        self.mode = 0

    def set_random_pos(self):
        random.seed = time.time()
        x = random.uniform(0, params['zone_scale'])
        y = random.uniform(0, params['zone_scale'])
        self.zone_coordi = [x, y]

    def set_zone_num(self):
        while True:
            zone_num = random.randrange(params['max_zone_num'])
            if zone_num not in [0, 49, 98, 147]:
                self.zone_num = zone_num
                break

    def get_abs_pos(self):
        self.zone_bp = [None, None]
        if 0 < self.zone_num <= 48:
            self.zone_bp = [-250.0, 240.0 - 10.0 * self.zone_num]
        elif 49 < self.zone_num <= 97:
            self.zone_bp = [-740.0 + 10.0 * self.zone_num, -250.0]
        elif 98 < self.zone_num <= 146:
            self.zone_bp = [240.0, -1230.0 + 10.0 * self.zone_num]
        elif 147 < self.zone_num <= 195:
            self.zone_bp = [1710.0 - 10.0 * self.zone_num, 240.0]
        else:
            assert "ERROR: zone_num is zero"
        return [self.zone_bp[i] + self.zone_coordi[i] for i in range(2)]
