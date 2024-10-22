import torch


class FourierDataset(torch.utils.data.Dataset):
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
        sin_series = torch.rand(self.scale, self.series, 1)
        cos_series = torch.rand(self.scale, self.series, 1)
        omega = torch.arange(self.series) * torch.pi
        # print(f'omega: {omega}, sin_series: {sin_series}, cos_series: {cos_series}')

        # u
        sin = sin_series * torch.sin(omega[:, None] * xc[None, :]).unsqueeze(0)
        cos = cos_series * torch.cos(omega[:, None] * xc[None, :]).unsqueeze(0)
        u = sin.sum(dim=1) + cos.sum(dim=1)
        
        # u_diff
        sin_diff = - cos_series * omega[None, :, None] * torch.sin(omega[:, None] * xc[None, :]).unsqueeze(0)
        cos_diff = sin_series * omega[None, :, None] * torch.cos(omega[:, None] * xc[None, :]).unsqueeze(0)
        u_diff = sin_diff.sum(dim=1) + cos_diff.sum(dim=1)

        # uv
        sin = sin_series * torch.sin(omega[:, None] * xv[None, :]).unsqueeze(0)
        cos = cos_series * torch.cos(omega[:, None] * xv[None, :]).unsqueeze(0)
        uv = sin.sum(dim=1) + cos.sum(dim=1)
        
        return u.unsqueeze(1), u_diff.unsqueeze(1), uv.unsqueeze(1)


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
