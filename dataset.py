import torch
import torch.utils
import h5py
import numpy as np


def get_dataset(name, config):
    if name == 'Fourier':
        return FourierDataset()
    elif name == 'Poly':
        return PolyDataset()
    elif name == 'Weno':
        return WenoDataset()
    elif name == 'Burgers1D':
        return Burgers1DDataset(config['data_path'])
    elif name == 'Advection1D':
        return Advection1DDataset(config['data_path'])
    elif name == 'ReactionDiffusion1D':
        return ReactionDiffusion1DDataset(config['data_path'])
    elif name == 'DiffusionSorption1D':
        return DiffusionSorption1DDataset(config['data_path'])
    elif name == 'ShallowWater2D':
        return ShallowWaterDataset(config['data_path'])
    else:
        raise ValueError(f'Unknown dataset {name}')

class Burgers1DDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        f = h5py.File(data_path, 'r')
        self.dt = 0.01
        self.dx = 2 / 1024
        self.nu = 0.001
        self.u = np.array(f['tensor'], dtype=np.float64)  # (10000, 201, 1024)
        self.u = torch.tensor(self.u, dtype=torch.float64)
        self.T = self.u.shape[1]
    
    def __len__(self):
        return self.u.shape[0]
    
    def __getitem__(self, idx):
        return self.u[idx]

class Advection1DDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        f = h5py.File(data_path, 'r')
        self.dt = 0.01
        self.dx = 1 / 1024
        self.u = np.array(f['tensor'], dtype=np.float64)
        self.u = torch.tensor(self.u, dtype=torch.float64)

    def __len__(self):
        return self.u.shape[0]
    
    def __getitem__(self, idx):
        return self.u[idx]

class ReactionDiffusion1DDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        f = h5py.File(data_path, 'r')
        self.dt = 0.01
        self.dx = 1 / 1024
        self.u = np.array(f['tensor'], dtype=np.float64)
        self.u = torch.tensor(self.u, dtype=torch.float64)

    def __len__(self):
        return self.u.shape[0]
    
    def __getitem__(self, idx):
        return self.u[idx]

class DiffusionSorption1DDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        f = h5py.File(data_path, 'r')
        self.dt = 0.01
        self.dx = 1 / 1024
        samples = []
        for i in range (10000):
            sample = np.array(f[str(i).zfill(4)]['data'], dtype=np.float64).squeeze()
            samples.append(sample)
        self.u = np.array(samples, dtype=np.float64)
        self.u = torch.tensor(self.u, dtype=torch.float64)  # (10000, 101, 1024)
    
    def __len__(self):
        return self.u.shape[0]
    
    def __getitem__(self, idx):
        return self.u[idx]

class ShallowWaterDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.f = h5py.File(data_path, 'r')
    
    def __len__(self):
        return len(self.f.keys())
    
    def __getitem__(self, idx):
        index = str(idx).zfill(4)
        data = np.array(self.f[str(idx)]['data'], dtype=np.float64)
        data = torch.tensor(data, dtype=torch.float64)
        return data  # (101, 128, 128, 3) h u v
    

class FourierDataset(torch.utils.data.Dataset):
    def __init__(self, num_cells, series, wavespeed, cfl_number, num_samples, dt, scale=1024):
        self.num_cells = num_cells
        self.series = series
        self.wavespeed = wavespeed
        self.cfl_number = cfl_number
        self.num_samples = num_samples
        self.dt = dt
        self.scale = scale
        self.nu = 0.001

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        xv = torch.linspace(-1, 1, self.num_cells + 1, dtype=torch.float64)
        xc = 0.5 * (xv[1:] + xv[:-1])
        sin_series = torch.rand(self.scale, self.series, 1) / self.series
        cos_series = torch.rand(self.scale, self.series, 1) / self.series
        omega = torch.arange(self.series) * torch.pi *2
        # print(f'omega: {omega}, sin_series: {sin_series}, cos_series: {cos_series}')

        # u
        sin = sin_series * torch.sin(omega[:, None] * xc[None, :]).unsqueeze(0)
        cos = cos_series * torch.cos(omega[:, None] * xc[None, :]).unsqueeze(0)
        u = sin.sum(dim=1) + cos.sum(dim=1)
        
        # u_diff
        sin_diff = - sin_series * omega[None, :, None]**2 * torch.sin(omega[:, None] * xc[None, :]).unsqueeze(0)
        cos_diff = - cos_series * omega[None, :, None]**2 * torch.cos(omega[:, None] * xc[None, :]).unsqueeze(0)
        u_diff = sin_diff.sum(dim=1) + cos_diff.sum(dim=1)

        # uv
        sin = sin_series * torch.sin(omega[:, None] * xv[None, :]).unsqueeze(0)
        cos = cos_series * torch.cos(omega[:, None] * xv[None, :]).unsqueeze(0)
        uv = sin.sum(dim=1) + cos.sum(dim=1)

        # unext
        xc_next = xc - self.wavespeed * self.dt
        xc_next = (xc_next + 1) % 2 - 1
        sin = sin_series * torch.sin(omega[:, None] * xc_next[None, :]).unsqueeze(0)
        cos = cos_series * torch.cos(omega[:, None] * xc_next[None, :]).unsqueeze(0)
        unext = sin.sum(dim=1) + cos.sum(dim=1)
        
        return u.unsqueeze(1), u_diff.unsqueeze(1), uv.unsqueeze(1), unext.unsqueeze(1)


class PDEBenchDataset(torch.utils.data.Dataset):
    def __init__(self, xc, numbers=10, k_tot=8, init_key=2022, num_k_chosen=2, norm=False):
        self.xc = xc
        self.numbers = numbers
        self.k_tot = k_tot
        self.init_key = init_key
        self.num_k_chosen = num_k_chosen
        self.nu = 0.001
    
        rng = np.random.default_rng(init_key)

        selected = rng.integers(0, k_tot, size=(numbers, num_k_chosen))
        selected = np.eye(k_tot)[selected].sum(axis=1)
        kk = 2.0 * np.pi * np.arange(1, k_tot + 1) * selected  / (xc[-1] - xc[0])
        amp = rng.uniform(size=(numbers, k_tot, 1))

        phs = 2.0 * np.pi * rng.uniform(size=(numbers, k_tot, 1))
        u = amp * np.sin(kk[:, :, np.newaxis] * xc[np.newaxis, np.newaxis, :] + phs)
        u = np.sum(u, axis=1)

        # Absolute value condition
        conds = rng.choice([0, 1], p=[0.9, 0.1], size=numbers)

        # Apply np.abs only where cond == 1
        u = np.where(conds[:, None] == 1, np.abs(u), u)

        # Random flip of sign
        sgn = rng.choice([1, -1], size=(numbers, 1))
        u *= sgn

        # Window function
        conds = rng.choice([0, 1], p=[0.9, 0.1], size=numbers)
        _xc = np.repeat(xc[None, :], numbers, axis=0)
        mask = np.ones_like(_xc)
        xL = rng.uniform(0.1, 0.45, size=(numbers, 1))  # shape (numbers, 1) for broadcasting
        xR = rng.uniform(0.55, 0.9, size=(numbers, 1))
        trns = 0.01 * np.ones_like(conds)[:, None]

        # Apply the window function where cond == 1
        mask = np.where(
            conds[:, None] == 1,
            0.5 * (np.tanh((_xc - xL) / trns) - np.tanh((_xc - xR) / trns)),
            mask
        )
        u *= mask

        if norm:
            u -= np.min(u, axis=1, keepdims=True)  # Positive values
            u /= np.max(u, axis=1, keepdims=True)  # Normalize to [0, 1]

        return u


class PolyDataset(torch.utils.data.Dataset):
    def __init__(self, num_cells, series, wavespeed, cfl_number, num_samples, scale=1024):
        self.num_cells = num_cells
        self.series = series
        self.wavespeed = wavespeed
        self.cfl_number = cfl_number
        self.num_samples = num_samples
        self.scale = scale

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        xv = torch.linspace(-1, 1, self.num_cells + 1)
        xc = 0.5 * (xv[1:] + xv[:-1])
        
        xc0 = torch.ones_like(xc)
        xc1 = xc
        xc2 = xc ** 2
        series = torch.stack([xc0, xc1, xc2], dim=1)
        series = series.unsqueeze(0).repeat(self.scale, 1, 1)
        coef0 = torch.rand(self.scale, 3)
        coef = coef0.unsqueeze(1).repeat(1, self.num_cells, 1)
        u = (series * coef).sum(dim=2)

        coef_diff = torch.stack([coef[:, :, 1], coef[:, :, 2]*2, torch.zeros_like(coef[:, :, 0])], dim=2)
        u_diff = (series * coef_diff).sum(dim=2)

        xv0 = torch.ones_like(xv)
        xv1 = xv
        xv2 = xv ** 2
        series = torch.stack([xv0, xv1, xv2], dim=1)
        series = series.unsqueeze(0).repeat(self.scale, 1, 1)
        coef = coef0.unsqueeze(1).repeat(1, self.num_cells + 1, 1)
        uv = (series * coef).sum(dim=2)

        
        return u, u_diff, uv


class WenoDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        u = torch.load(f'dataset/u{idx}.pt')
        uv = torch.load(f'dataset/uv{idx}.pt')
        return u, u, uv




if __name__ == '__main__':
    num_cells = 1024
    xv = torch.linspace(-1, 1, 1024 + 1)
    xc = 0.5 * (xv[1:] + xv[:-1])
    dataset = PDEBenchDataset(xc)

