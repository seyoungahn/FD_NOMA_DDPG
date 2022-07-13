# -*- coding: utf-8 -*-
import random

# System configurations
import json
with open('params.json', 'r') as f:
    params = json.load(f)

RANDOM_SEED_LIST = random.sample(range(1000), 500)
ZONE_QUEUE = [0 for _ in range(params['n_zone'])]
EPOCHES = 0