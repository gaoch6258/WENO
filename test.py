from model import stencilCNN, PDE
import torch
from matplotlib import pyplot as plt
import time
from dataset import FourierDataset, PolyDataset, WenoDataset, BurgersDataset
import numpy as np

torch.set_printoptions(precision=20)

cfl_number = 0.5
wavespeed = 1
num_cells = 200
xv = torch.linspace(0, 1, num_cells + 1).double()
xc = 0.5 * (xv[1:] + xv[:-1])
dx = 2/1024
dt = 1e-3

print('------- Reading Dataset --------------')
dataset = BurgersDataset()
# dataset = FourierDataset(1024, 16, 1, 0.5, 10000, dt, scale=32)
print('------- Dataset loaded --------------')
# dataset = WenoDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

net = stencilCNN(dt, dx).cuda()
# net.load_state_dict(torch.load('./ckpt/stencilCNN.pt'))
net.eval()
criterion = torch.nn.MSELoss()
pde = PDE(net, 'Burgers1D', {'nu': 0})


## test arena #########################################
# xv = np.linspace(0, 1, num_cells + 1, dtype=np.float64)
# xc = 0.5 * (xv[1:] + xv[:-1])
# u = np.sin(2 * 3.1415926535 * xc)
# u = torch.tensor(u, dtype=torch.float64)
# u = u.unsqueeze(0).unsqueeze(0).cuda()
# # du = pde.RK4(u)
# for i in range(10000):
#     print(i)
#     rhs = net.flux_splitting(u)
#     ut = u + dt * rhs
#     rhs = net.flux_splitting(ut)
#     ut = 0.75 * u + 0.25 * (ut + dt * rhs)
#     rhs = net.flux_splitting(ut)
#     u = (u + 2 * (ut + dt * rhs)) / 3
#     if i%1000 == 0:
#         plt.plot(u[0, 0, :].detach().cpu().numpy())
#         plt.savefig(f'./figs/{i}.png')
#         plt.close()
# exit()
## test arena end #####################################

# optimizer = torch.optim.Adam(net.train_params(), lr=1e-3)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 45 * 100 )

# def nan_hook(self, inp, output):
#     if not isinstance(output, tuple):
#         outputs = [output]
#     else:
#         outputs = output

#     for i, out in enumerate(outputs):
#         nan_mask = torch.isnan(out)
#         if nan_mask.any():
#             print("In", self.__class__.__name__)
#             raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

# for submodule in net.modules():
#     submodule.register_forward_hook(nan_hook)

mini_start = start = time.time()
for i, data in enumerate(dataloader):
    u, u_diff, uv, u_next = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda()
    B, S, C, L = u.shape
    u, u_diff, uv, u_next = u.reshape(B * S, 1, L), u_diff.reshape(B * S, 1, L), uv.reshape(B * S, -1, L+1), u_next.reshape(B * S, 1, L)
    # print(f'Data Time: {time.time() - mini_start}')
    # num_cells = 2000
    mini_start = time.time()
    B, T, L = u.shape
    # print(u.shape)
    # xv = torch.linspace(-1, 1, num_cells + 1, dtype=torch.float64)
    # xc = 0.5 * (xv[1:] + xv[:-1])
    # u = xc**2
    # u = u.unsqueeze(0).unsqueeze(0).cuda()
    
    outputs = []
    uc = u[:, 0:1, :]
    for j in range(10):
        for k in range(500):
            with torch.no_grad():
                # k1 = dataset.nu / torch.pi * net.weno(net.weno(uc)/net.dt) - net.weno(uc) * uc
                # # k1 = - net.weno(uc)
                # k2 = dataset.nu / torch.pi * net.weno(net.weno(uc + k1*0.5)/net.dt) - net.weno(uc + k1*0.5) * (uc + k1*0.5) 
                # k3 = dataset.nu / torch.pi * net.weno(net.weno(uc + k2*0.5)/net.dt) - net.weno(uc + k2*0.5) * (uc + k2*0.5) 
                # k4 = dataset.nu / torch.pi * net.weno(net.weno(uc + k3*1)/net.dt) - net.weno(uc + k3*1) * (uc + k3*1) 

                # k1 = - net.weno(uc) * uc
                # k2 = - net.weno(uc + k1*0.5) * (uc + k1*0.5)
                # k3 = - net.weno(uc + k2*0.5) * (uc + k2*0.5)
                # k4 = - net.weno(uc + k3*1) * (uc + k3*1)
                # print(torch.max(torch.abs(k1)), torch.max(torch.abs(k2)), torch.max(torch.abs(k3)), torch.max(torch.abs(k4)))
                # du = net.fc(torch.stack((k1, k2, k3, k4), dim=-1)).squeeze(-1)
                # du = (k1 + 2*k2 + 2*k3 + k4) / 6
                # du = k1
                du = pde.RK4(uc)
                uc = uc + du
        # plt.plot(u[0, j+1, :256].detach().cpu().numpy(), 'r', label='gt')
        print(uc[0, 0, :].detach().cpu().numpy())
        plt.plot(uc[0, 0, :].detach().cpu().numpy(), 'b', label='next')
        plt.legend()
        plt.savefig(f'./figs/{i}_{j}.png')
        plt.close()
        outputs.append(uc)
    output = torch.stack(outputs, dim=1).squeeze(2)
    loss = torch.sqrt(criterion(output, u[:, 1:]))
    if i%20 == 0:
        print(f'Iter {i}, Loss: {loss.item()}')


