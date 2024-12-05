import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py

f = h5py.File('/home/gaoch/dataset/diffusionsorption1D/1D_diff-sorp_NA_NA.h5', 'r')
print(list(f.keys())[:100])
# t = np.array(f['t-coordinate'])
# x = np.array(f['x-coordinate'])
u = f['0001']
# nu = np.array(f['nu'])
print(u)
print(type(u[0]))
print(u[1])

print(f'U shape: {u.shape}, t shape: {t.shape}, x shape: {x.shape}')
diffx = np.diff(x)
difft = np.diff(t)
# print(f'diffx: {diffx}, difft: {difft}')
# print(f'x: {x}, t: {t}')
print(x[-1] - x[0])





