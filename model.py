import torch 
import torch.nn as nn
from matplotlib import pyplot as plt
import time

def stencil_a(f0, f1, f2, f3, f4): return  2/6 * f0 - 7/6 * f1 + 11/6 * f2
def stencil_b(f0, f1, f2, f3, f4): return -1/6 * f1 + 5/6 * f2 +  2/6 * f3
def stencil_c(f0, f1, f2, f3, f4): return  1/3 * f2 + 5/6 * f3 -  1/6 * f4


class stencilCNN(nn.Module):
    def __init__(self, dt, dx):
        super(stencilCNN, self).__init__()
        # self.stencil_1 = nn.Parameter(torch.tensor([2/6, -7/6, 11/6], dtype=torch.float32))
        # self.stencil_2 = nn.Parameter(torch.tensor([-1/6, 5/6, 2/6], dtype=torch.float32))
        # self.stencil_3 = nn.Parameter(torch.tensor([1/3, 5/6, -1/6], dtype=torch.float32))
        
        # self.stencil = nn.Conv1d(1, 3, 5, padding=0, bias=False)
        self.stencil = torch.nn.Parameter(torch.tensor([[[2/6, -7/6, 11/6, 0, 0]], [[0, -1/6, 5/6, 2/6, 0]], [[0, 0, 1/3, 5/6, -1/6]]], dtype=torch.float32))
        self.stencil = torch.nn.Parameter(torch.tensor([[[1/3]], [[-1/6]], [[1/3]]], dtype=torch.float32))
        # self.stencil = torch.nn.Parameter(torch.tensor([[[-0.3690,  0.2299,  1.1310, 0, 0]], [[0, 0.0630,  0.3766,  0.5630, 0]], [[0, 0, 0.5630,  0.3766,  0.0630]]], dtype=torch.float32))
        # self.stencil.weight = torch.nn.Parameter(torch.tensor([[[0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0]]], dtype=torch.float32))
        # self.stencil.weight = torch.nn.Parameter(torch.tensor([[[2/6, -7/6, 11/6, 0, 0]], [[0, -1/6, 5/6, 2/6, 0]], [[0, 0, 1/3, 5/6, -1/6]]], dtype=torch.float32))
        # self.stencil.weight = torch.nn.Parameter(torch.tensor([[[0, 0, 3/8, 6/8, -1/8]], [[0, 0, 3/8, 6/8, -1/8]], [[0, 0, 3/8, 6/8, -1/8]]], dtype=torch.float32))
        
        # self.stencil.bias = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))
        # self.mask = torch.tensor([[[1, 1, 1, 0, 0]], [[0, 1, 1, 1, 0]], [[0, 0, 1, 1, 1]]], dtype=torch.float32).cuda()
        # for p in self.stencil.parameters():
        #     p.requires_grad = False
        self.k = torch.nn.Parameter(torch.tensor([0.5, 0.5, 1], dtype=torch.float32))
        self.fc = nn.Linear(4, 1, bias=False)
        self.weno_flag = False
        self.dt = dt
        self.dx = dx
        # self.fc = nn.Linear(3, 1)
        # self.fc.weight = torch.nn.Parameter(torch.tensor([[1/3, 1/3, 1/3]], dtype=torch.float32))
        # self.fc.bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        # for p in self.fc.parameters():
        #     p.requires_grad = False
        # self.weno = nn.ModuleList([self.stencil, self.fc])
    def betas(self, u):
        u00 = u[:, :, :-5] * u[:, :, :-5]
        u01 = u[:, :, :-5] * u[:, :, 1:-4]
        u11 = u[:, :, 1:-4] * u[:, :, 1:-4]
        u02 = u[:, :, :-5] * u[:, :, 2:-3]
        u12 = u[:, :, 1:-4] * u[:, :, 2:-3]
        u22 = u[:, :, 2:-3] * u[:, :, 2:-3]
        u13 = u[:, :, 1:-4] * u[:, :, 3:-2]
        u23 = u[:, :, 2:-3] * u[:, :, 3:-2]
        u33 = u[:, :, 3:-2] * u[:, :, 3:-2]
        u24 = u[:, :, 2:-3] * u[:, :, 4:-1]
        u34 = u[:, :, 3:-2] * u[:, :, 4:-1]
        u44 = u[:, :, 4:-1] * u[:, :, 4:-1]
        beta_1 = (1./3. * (4.*u00 - 19.*u01 + 25.*u11 + 11.*u02 - 31.*u12 + 10.*u22))
        beta_2 = (1./3. * (4.*u11 - 13.*u12 + 13.*u22 + 5.*u13 - 13.*u23 + 4.*u33))
        beta_3 = (1./3. * (10.*u22 - 31.*u23 + 25.*u33 + 11.*u24 - 19.*u34 + 4.*u44))
        return beta_1, beta_2, beta_3
    
    def w(self, beta1, beta2, beta3):
        gammas = [1./10., 6./10., 3./10.]
        eps = 1.e-6
        w_til_1 = gammas[0]/(eps+beta1)**2
        w_til_2 = gammas[1]/(eps+beta2)**2
        w_til_3 = gammas[2]/(eps+beta3)**2

        w_til = w_til_1 + w_til_2 + w_til_3

        w1 = w_til_1/w_til
        w2 = w_til_2/w_til
        w3 = w_til_3/w_til
        return w1, w2, w3
    
    def weno(self, u):
        u = nn.functional.pad(u, (3, 3), mode='circular')
        # self.stencil.weight.data = self.stencil.weight.data * self.mask
        weight1 = torch.cat((self.stencil[0], -torch.ones(1, 1).cuda() * 0.5 - 2 * self.stencil[0], torch.ones(1, 1).cuda() * 1.5 + self.stencil[0], torch.zeros(1, 2).cuda()), dim=-1)
        weight2 = torch.cat((torch.zeros(1, 1).cuda(), self.stencil[1], torch.ones(1, 1).cuda() * 0.5 - 2 * self.stencil[1], 
                             torch.ones(1, 1).cuda() * 0.5 + self.stencil[1], torch.zeros(1, 1).cuda()), dim=-1)
        weight3 = torch.cat((torch.zeros(1, 2).cuda(), self.stencil[2], torch.ones(1, 1).cuda() * 1.5 - 2 * self.stencil[2], -torch.ones(1, 1).cuda() * 0.5 + self.stencil[2]), dim=-1)
        weight = torch.stack([weight1, weight2, weight3], dim=0)
        k = nn.functional.conv1d(u, weight, padding=0, bias=None)[:, :, :-1]
        # k = self.stencil(u)[:, :, :-1]
        if self.weno_flag:
            beta1, beta2, beta3 = self.betas(u)
            w1, w2, w3 = self.w(beta1, beta2, beta3)
            k = w1 * k[:, 0:1] + w2 * k[:, 1:2] + w3 * k[:, 2:3]
            # k[:, 0:1] *= w1
            # k[:, 1:1] *= w2
            # k[:, 2:1] *= w3
            # return (k[:, :, :-1] - k[:, :, 1:]) * self.dt / self.dx
            return k
        else:
            k = k.mean(dim=1, keepdim=True)
            return (k[:, :, :-1] - k[:, :, 1:]) * self.dt / self.dx
            return k
    
    # def forward(self, u):
    #     k1 = self.weno(u)
    #     k2 = self.weno(u + k1 * 0.5)
    #     k3 = self.weno(u + k2 * 0.5)
    #     k4 = self.weno(u + k3 * 1.0)
    #     u += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    #     return u

    # def forward(self, u):
    #     k1 = self.weno(u)
    #     k2 = self.weno(u + k1 * self.k[0])
    #     k3 = self.weno(u + k2 * self.k[1])
    #     k4 = self.weno(u + k3 * self.k[2])
    #     u += self.fc(torch.stack((k1, k2, k3, k4), dim=-1)).squeeze()
    #     return u
    def forward(self, u):
        return self.weno(u)
    
    def train_params(self):
        return self.parameters()


def gt_generator(name, wavespeed, xc, t):
    # periodic 
    xc = xc - wavespeed * t
    xc = (xc + 1) % 2 - 1
    if name == 'sin':
        return torch.sin(torch.pi * xc)
    if name == 'square':
        return 0.5 + 0.5 * torch.sign(torch.sin(2 * torch.pi * xc))
    if name == 'sawtooth':
        return 0.5 + 0.5 * (xc - wavespeed * t - torch.floor(xc))
    if name == 'triangle':
        return 0.5 + 0.5 * (1 - 2 * torch.abs(xc - torch.floor(xc) - 0.5))
    if name == 'gaussian':
        return torch.exp(-100 * (xc)**2)
    if name == 'step':
        return 0.5 + 0.5 * torch.sign(xc)
    if name == 'ramp':
        return 0.5 + 0.5 * (xc)



if __name__ == '__main__':
    cfl_number = 0.5
    wavespeed = 1
    num_cells = 128
    name = 'step'
    net = stencilCNN()
    xv = torch.linspace(-1, 1, num_cells + 1)
    xc = 0.5 * (xv[1:] + xv[:-1])
    dx = xv[1] - xv[0]
    dt = abs(cfl_number * dx.item() / wavespeed)
    u = gt_generator(name, wavespeed, xc, 0)[None, None, :]
    t = 0.0
    for i in range(100):
        u = net(u)
        t += dt
        u_gt = gt_generator(name, wavespeed, xc, t)
        err = ((u - u_gt)**2).mean().item()
        plt.plot(u[0, 0].detach().numpy(), '.', markersize='5', label = "Numeric, t=%.2f" %i, zorder = 0)
        plt.plot(u_gt.detach().numpy())
        plt.savefig(f'./figs/{i}.png')
        plt.close()
        print(f'err at time {t:.2f}: {err:.2e}')

