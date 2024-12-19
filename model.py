import torch 
import torch.nn as nn
from matplotlib import pyplot as plt
import time

def stencil_a(f0, f1, f2, f3, f4): return  2/6 * f0 - 7/6 * f1 + 11/6 * f2
def stencil_b(f0, f1, f2, f3, f4): return -1/6 * f1 + 5/6 * f2 +  2/6 * f3
def stencil_c(f0, f1, f2, f3, f4): return  1/3 * f2 + 5/6 * f3 -  1/6 * f4


class stencilCNN1D(nn.Module):
    def __init__(self, dt, dx, name, params):
        super(stencilCNN1D, self).__init__()
        self.stencil = torch.nn.Parameter(torch.tensor([[[1/3]], [[-1/6]], [[1/3]]], dtype=torch.float64))
        # self.stencil.weight = torch.nn.Parameter(torch.tensor([[[2/6, -7/6, 11/6, 0, 0]], [[0, -1/6, 5/6, 2/6, 0]], [[0, 0, 1/3, 5/6, -1/6]]], dtype=torch.float64))
        # self.stencil.weight = torch.nn.Parameter(torch.tensor([[[0, 0, 3/8, 6/8, -1/8]], [[0, 0, 3/8, 6/8, -1/8]], [[0, 0, 3/8, 6/8, -1/8]]], dtype=torch.float64))
        
        self.k = torch.nn.Parameter(torch.tensor([0.5, 0.5, 1], dtype=torch.float64))
        self.fc = nn.Linear(4, 1, bias=False)
        self.fc.weight = torch.nn.Parameter(torch.tensor([[1/6, 2/6, 2/6, 1/6]], dtype=torch.float64))
        self.pooling = nn.MaxPool1d(5, stride=1, padding=0)
        self.weno_flag = True
        self.dt = dt
        self.dx = dx
        self.name = name
        self.params = params

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
        eps = 1.e-10
        w_til_1 = gammas[0]/(eps+beta1)**2
        w_til_2 = gammas[1]/(eps+beta2)**2
        w_til_3 = gammas[2]/(eps+beta3)**2

        w_til = w_til_1 + w_til_2 + w_til_3

        w1 = w_til_1/w_til
        w2 = w_til_2/w_til
        w3 = w_til_3/w_til
        return w1, w2, w3
    
    def upwind(self, u):
        u = nn.functional.pad(u, (1, 1), mode='circular')
        uc = u[:, :, 2:] - u[:, :, :-2]
        return uc / self.dx / 2
    
    def upwind2(self, u, bc='periodic'):
        if bc == 'periodic':
            u = nn.functional.pad(u, (2, 2), mode='circular')
            u1 = (u[:, :, 2:] - u[:, :, 1:-1]) / self.dx
            u2 = (u1[:, :, 1:-1] - u1[:, :, :-2]) / self.dx
        elif bc == 'second':
            pad = u[:, :, 0:1] * 0
            u = torch.cat((pad, u, pad), dim=-1)
            u2 = u[:, :, 2:] - 2 * u[:, :, 1:-1] + u[:, :, :-2]
            u2 /= self.dx**2
        # uc = -u[:, :, 4:] + 4 * u[:, :, 3:-1] - 6 * u[:, :, 2:-2] + 4 * u[:, :, 1:-3] - u[:, :, :-4]
        # return uc / self.dx / 12

        return u2
    
    def weno2(self, u):
        return self.weno(self.weno(u))
    
    def weno_x(self, u):
        B, T, L, W = u.shape
        res = self.weno(torch.permute(u, (0, 2, 1, 3)).reshape(-1, T, W))
        return res.reshape(B, L, T, W).permute(0, 2, 1, 3)
    
    def weno_y(self, u):
        B, T, L, W = u.shape
        res = self.weno(torch.permute(u, (0, 3, 1, 2)).reshape(-1, T, L))
        return res.reshape(B, W, T, L).permute(0, 2, 3, 1)

    
    def _apply_bc(self, u):
        if self.params['bc'] == 'periodic':
            return torch.concat((u[:, :, -4:-1], u, u[:, :, 1:4]), dim=-1)
        elif self.params['bc'] == 'second':
            return nn.functional.pad(u, (3, 3), mode='replicate')

    def weno(self, u):
        u = self._apply_bc(u)
        # u = torch.concat((u[:, :, -3:], u, u[:, :, :3]), dim=-1)
        # self.stencil.weight.data = self.stencil.weight.data * self.mask
        weight1 = torch.cat((self.stencil[0], -torch.ones(1, 1).cuda() * 0.5 - 2 * self.stencil[0], torch.ones(1, 1).cuda() * 1.5 + self.stencil[0], torch.zeros(1, 2).cuda()), dim=-1)
        weight2 = torch.cat((torch.zeros(1, 1).cuda(), self.stencil[1], torch.ones(1, 1).cuda() * 0.5 - 2 * self.stencil[1], 
                             torch.ones(1, 1).cuda() * 0.5 + self.stencil[1], torch.zeros(1, 1).cuda()), dim=-1)
        weight3 = torch.cat((torch.zeros(1, 2).cuda(), self.stencil[2], torch.ones(1, 1).cuda() * 1.5 - 2 * self.stencil[2], -torch.ones(1, 1).cuda() * 0.5 + self.stencil[2]), dim=-1)
        weight = torch.stack([weight1, weight2, weight3], dim=0)

        k = nn.functional.conv1d(u, weight, padding=0, bias=None)[:, :, :-1]  # [B, 3, L]
        if self.weno_flag:
            # beta1, beta2, beta3 = self.betas(u)
            # w1, w2, w3 = self.w(beta1, beta2, beta3)
            s1 = 13/12 * (u[:, :, :-5] - 2 * u[:, :, 1:-4] + u[:, :, 2:-3])**2 + 1/4 * (u[:, :, :-5] - 4 * u[:, :, 1:-4] + 3 * u[:, :, 2:-3])**2
            s2 = 13/12 * (u[:, :, 1:-4] - 2 * u[:, :, 2:-3] + u[:, :, 3:-2])**2 + 1/4 * (u[:, :, 1:-4] - u[:, :, 3:-2])**2
            s3 = 13/12 * (u[:, :, 2:-3] - 2 * u[:, :, 3:-2] + u[:, :, 4:-1])**2 + 1/4 * (3 * u[:, :, 2:-3] - 4 * u[:, :, 3:-2] + u[:, :, 4:-1])**2

            eps = 1e-12
            a1 = 1/10 / (eps + s1)**2
            a2 = 6/10 / (eps + s2)**2
            a3 = 3/10 / (eps + s3)**2

            w1 = a1 / (a1 + a2 + a3)
            w2 = a2 / (a1 + a2 + a3)
            w3 = a3 / (a1 + a2 + a3)
            
            k = w1 * k[:, 0:1] + w2 * k[:, 1:2] + w3 * k[:, 2:3]

            return (k[:, :, 1:] - k[:, :, :-1]) / self.dx
            return k
        else:
            k = k.mean(dim=1, keepdim=True)
            return (k[:, :, 1:] - k[:, :, :-1]) / self.dx
            return k
    
    def flux_splitting(self, u):
        f = 0.5 * u * u
        u_pad = nn.functional.pad(u, (2, 2), mode='circular')
        umax = self.pooling(torch.abs(u_pad))
        fp = 0.5 * (f + umax * u)
        fm = 0.5 * (f - umax * u)

        # print('u: ', u[:,:,:12])
        # print('f: ', f[:,:,:11])
        # print('umax: ', umax[:,:,:11])
        # print('fp: ', fp[:,:,:12])
        # print('fm: ', fm[:,:,:12])
        wenoL = self.weno(fp)
        # print('WENOL:', wenoL[:, :, :10])
        wenoR = torch.flip(-self.weno(torch.flip(fm, dims=[-1])), dims=[-1])
        # print(wenoR[:, :, :10])
        return -wenoL - wenoR

    def forward(self, uc):
        if self.name == 'Burgers1D':
            k1 = self.params['nu'] / torch.pi * self.weno(self.upwind(uc)) * self.dt + self.flux_splitting(uc) * self.dt
            k2 = self.params['nu'] / torch.pi * self.weno(self.upwind(uc + k1*0.5)) * self.dt + self.flux_splitting(uc + k1*0.5) * self.dt
            k3 = self.params['nu'] / torch.pi * self.weno(self.upwind(uc + k2*0.5)) * self.dt + self.flux_splitting(uc + k2*0.5) * self.dt
            k4 = self.params['nu'] / torch.pi * self.weno(self.upwind(uc + k3*1)) * self.dt + self.flux_splitting(uc + k3*1) * self.dt

            du = (k1 + 2*k2 + 2*k3 + k4) / 6
            return du
        
        elif self.name == 'Advection1D':
            k1 = - self.params['beta'] * self.weno(uc) * self.dt
            k2 = - self.params['beta'] * self.weno(uc + k1*0.5) * self.dt
            k3 = - self.params['beta'] * self.weno(uc + k2*0.5) * self.dt
            k4 = - self.params['beta'] * self.weno(uc + k3*1) * self.dt
            du = (k1 + 2*k2 + 2*k3 + k4) / 6
            return du

        elif self.name == 'ReactionDiffusion1D':
            rho = self.params['rho']
            nu = self.params['nu']

            k1 = nu * self.upwind2(uc) * self.dt
            # k2 = nu * self.model.weno2(uc + k1 * 0.5) * self.model.dt
            # k3 = nu * self.model.weno2(uc + k2 * 0.5) * self.model.dt
            # k4 = nu * self.model.weno2(uc + k3) * self.model.dt
            # du = (k1 + 2*k2 + 2*k3 + k4) / 6
            du = k1
            tmp = torch.exp(-torch.ones(1, dtype=torch.float64)*self.params['rho']*self.dt*0.5).item()
            up = 1 / (1 + tmp * (1-uc) / uc)

            utmp = up + du * 0.5
            k2 = nu * self.upwind2(utmp) * self.dt
            tmp = torch.exp(-torch.ones(1, dtype=torch.float64)*self.params['rho']*self.dt).item()
            up = 1 / (1 + tmp * (1-uc) / uc)
            u = up + k2
            du = u - uc
            return du

        elif self.name == 'DiffusionSorption1D':
            # print(self.params)
            phi, nf, D = self.params['phi'], self.params['nf'], self.params['D']
            retard = 1 + (1 - phi) / phi * self.params['rho_s'] * self.params['k'] * nf * uc ** (nf-1)
            du = D / retard * self.upwind2(uc, bc='second') * self.dt
            du[:, :, 0] += D / retard[:, :, 0] / self.dx ** 2 * self.dt
            du[:, :, -1] += D / retard[:, :, -1] / self.dx**2 * (uc[:,:,-2] - uc[:,:,-1]) / self.dx * D * self.dt
            return du
        
        elif self.name == 'ShallowWater2D':
            g = self.params['g']
            h = uc[..., 0]
            u = uc[..., 1]
            v = uc[..., 2]
            hu = h * u
            hv = h * v

            k1_h = - self.weno_x(hu) * self.dt - self.weno_y(hv) * self.dt
            k1_hu = - self.weno_x(hu*u + 0.5*g*h*h) * self.dt - self.weno_y(hu*v)
            k1_hv = - self.weno_y(hv*v + 0.5*g*h*h) * self.dt - self.weno_x(hu*v)

            h_new = h + k1_h
            u_new = (hu + k1_hu) / h_new
            v_new = (hv + k1_hv) / h_new
            return torch.stack((h_new, u_new, v_new), dim=-1)
        
        elif self.name=='Advection2D':
            vx = self.param['vx']
            vy = self.param['vy']
            k1_x = - vx * self.weno_x(uc) * self.dt
            k2_x = - vx * self.weno_x(uc + k1_x*0.5) * self.dt
            k3_x = - vx * self.weno_x(uc + k2_x*0.5) * self.dt
            k4_x = - vx * self.weno_x(uc + k3_x*1) * self.dt

            k1_y = - vy * self.weno_y(uc) * self.dt
            k2_y = - vy * self.weno_y(uc + k1_y*0.5) * self.dt
            k3_y = - vy * self.weno_y(uc + k2_y*0.5) * self.dt
            k4_y = - vy * self.weno_y(uc + k3_y*1) * self.dt

            du = (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6 + (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
            return du
        
        elif self.name=='ReactionDiffusion2D':
            Du = self.param['Du']
            Dv = self.param['Dv']
            k = self.param['k']

            k1x = Du * self.upwind2(uc[..., 0], bc='second') * self.dt
            k1y = Du * self.upwind2(uc[..., 0], bc='second') * self.dt
            k2x = Du * self.upwind2(uc[..., 0] + k1x*0.5, bc='second') * self.dt
            k2y = Du * self.upwind2(uc[..., 0] + k1y*0.5, bc='second') * self.dt

            dux = - k2x + (u-u**3-k-v) * self.dt
            duy = - k2y + (u-v) * self.dt 
            return torch.stack((dux, duy), dim=-1)
            
            
        else:
            raise NotImplementedError(f"Unknown PDE: {self.name}")
    
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


class PDE():
    def __init__(self, model: stencilCNN1D, name='Advection1D', bc='periodic', params=None):
        self.name = name
        self.model = model
        self.params = params
        self.bc = bc
    
    def RK4(self, uc):
        net = self.model
        if self.name == 'Burgers1D':
            k1 = self.params['nu'] / torch.pi * net.weno(net.upwind(uc)) * net.dt + net.flux_splitting(uc) * net.dt
            k2 = self.params['nu'] / torch.pi * net.weno(net.upwind(uc + k1*0.5)) * net.dt + net.flux_splitting(uc + k1*0.5) * net.dt
            k3 = self.params['nu'] / torch.pi * net.weno(net.upwind(uc + k2*0.5)) * net.dt + net.flux_splitting(uc + k2*0.5) * net.dt
            k4 = self.params['nu'] / torch.pi * net.weno(net.upwind(uc + k3*1)) * net.dt + net.flux_splitting(uc + k3*1) * net.dt

            du = (k1 + 2*k2 + 2*k3 + k4) / 6
            return du
        
        elif self.name == 'Advection1D':
            k1 = - self.params['beta'] * self.model.weno(uc) * self.model.dt
            k2 = - self.params['beta'] * self.model.weno(uc + k1*0.5) * self.model.dt
            k3 = - self.params['beta'] * self.model.weno(uc + k2*0.5) * self.model.dt
            k4 = - self.params['beta'] * self.model.weno(uc + k3*1) * self.model.dt
            du = (k1 + 2*k2 + 2*k3 + k4) / 6
            return du

        elif self.name == 'ReactionDiffusion1D':
            rho = self.params['rho']
            nu = self.params['nu']

            k1 = nu * self.model.upwind2(uc) * self.model.dt
            # k2 = nu * self.model.weno2(uc + k1 * 0.5) * self.model.dt
            # k3 = nu * self.model.weno2(uc + k2 * 0.5) * self.model.dt
            # k4 = nu * self.model.weno2(uc + k3) * self.model.dt
            # du = (k1 + 2*k2 + 2*k3 + k4) / 6
            du = k1
            tmp = torch.exp(-torch.ones(1, dtype=torch.float64)*self.params['rho']*self.model.dt*0.5).item()
            up = 1 / (1 + tmp * (1-uc) / uc)

            utmp = up + du * 0.5
            k2 = nu * self.model.upwind2(utmp) * self.model.dt
            tmp = torch.exp(-torch.ones(1, dtype=torch.float64)*self.params['rho']*self.model.dt).item()
            up = 1 / (1 + tmp * (1-uc) / uc)
            u = up + k2
            du = u - uc
            return du

        elif self.name == 'DiffusionSorption1D':
            # print(self.params)
            phi, nf, D = self.params['phi'], self.params['nf'], self.params['D']
            retard = 1 + (1 - phi) / phi * self.params['rho_s'] * self.params['k'] * nf * uc ** (nf-1)
            du = D / retard * self.model.upwind2(uc, bc='second') * self.model.dt
            du[:, :, 0] += D / retard[:, :, 0] / self.model.dx ** 2 * self.model.dt
            du[:, :, -1] += D / retard[:, :, -1] / self.model.dx**2 * (uc[:,:,-2] - uc[:,:,-1]) / self.model.dx * D * self.model.dt
            return du
        
        elif self.name == 'ShallowWater2D':
            g = self.params['g']
            h = uc[..., 0]
            u = uc[..., 1]
            v = uc[..., 2]
            hu = h * u
            hv = h * v

            k1_h = self.weno_x(hu)
        else:
            raise NotImplementedError(f"Unknown PDE: {self.name}")
