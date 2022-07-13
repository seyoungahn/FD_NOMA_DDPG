# def _get_ACK(idx):
#     ptr = 0
#     max_val = idx[0]
#     ACK = [0, 0, 0, 0]
#     for i in range(4):
#         if idx[i] == ptr:
#             ACK[i] = 1
#             ptr += 1
#         else:
#             if ptr >= max_val:
#                 ACK[i] = 1
#                 ptr += 1
#                 max_val = idx[i]
#     # result = [0, 0, 0, 0]
#     # for i in range(4):
#     #     result[(idx[i]+3)%4] = ACK[i]
#     return ACK
#
# power = [[1, 2, 3, 4],
#          [1, 2, 4, 3],
#          [1, 3, 2, 4],
#          [1, 3, 4, 2],
#          [1, 4, 2, 3],
#          [1, 4, 3, 2],
#          [2, 1, 3, 4],
#          [2, 1, 4, 3],
#          [2, 3, 1, 4],
#          [2, 3, 4, 1],
#          [2, 4, 1, 3],
#          [2, 4, 3, 1],
#          [3, 1, 2, 4],
#          [3, 1, 4, 2],
#          [3, 2, 1, 4],
#          [3, 2, 4, 1],
#          [3, 4, 1, 2],
#          [3, 4, 2, 1],
#          [4, 1, 2, 3],
#          [4, 1, 3, 2],
#          [4, 2, 1, 3],
#          [4, 2, 3, 1],
#          [4, 3, 1, 2],
#          [4, 3, 2, 1]]
#
# for elem in power:
#     for i in range(4):
#         elem[i] -= 1
#
# for elem in power:
#     print(_get_ACK(elem))

import json
import utils

with open('params.json', 'r') as f:
    params = json.load(f)

noise = utils.dBm2mW(params['white_noise']) * params["BW"] * 10e+6

print(utils.mW2dBm((10 * noise) / utils.dBm2mW(utils.pathloss(20.0, 'V2V', 'LOS'))))