import numpy as np
from itertools import product

vec = np.array([2.148078,-1.466622,-0.696094,0.067750])

min_val = -1.5
max_val = 2.5

total_bits = 20
dim = len(vec)

# 枚举所有分配方式（可能多，注意时间复杂度）
# b_i 从1到16，和为20
bit_ranges = range(1, 17)
best_error = float('inf')
best_alloc = None
best_reconstruction = None

for bits_alloc in product(bit_ranges, repeat=dim):
    if sum(bits_alloc) == total_bits:
        levels = [2**b - 1 for b in bits_alloc]
        deltas = [(max_val - min_val) / l for l in levels]
        q = [np.clip(round((v - min_val) / d), 0, l) for v, d, l in zip(vec, deltas, levels)]
        recon = np.array([qi * di + min_val for qi, di in zip(q, deltas)])
        error = np.mean((vec - recon) ** 2)  # MSE
        if error < best_error:
            best_error = error
            best_alloc = bits_alloc
            best_reconstruction = recon
            best_bitstring = ''.join([format(x, f'0{b}b') for x, b in zip(q, bits_alloc)])

print("最优bit分配:", best_alloc)
print("重构误差MSE:", best_error)
print("重构向量:", best_reconstruction)
print("对应二进制编码序列:", best_bitstring)
print("二进制长度:", len(best_bitstring))