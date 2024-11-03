from model import stencilCNN
import torch
from matplotlib import pyplot as plt
import time
from dataset import FourierDataset, PolyDataset, WenoDataset

cfl_number = 0.5
wavespeed = 1
num_cells = 512
xv = torch.linspace(-1, 1, num_cells + 1)
xc = 0.5 * (xv[1:] + xv[:-1])
dx = xv[1] - xv[0]
dt = abs(cfl_number * dx.item() / wavespeed)
resume = True

dataset = FourierDataset(num_cells=num_cells, series=16, wavespeed=1, cfl_number=0.5, num_samples=81920, dt=dt, scale=512)
# dataset = WenoDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

net = stencilCNN(dt, dx).cuda()
if resume:
    net.load_state_dict(torch.load('./ckpt/stencilCNN.pt'))
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.train_params(), lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 45 * 100 )

for epoch in range(10000):
    mini_start = start = time.time()
    for i, data in enumerate(dataloader):
        # print(f'Data Time: {time.time() - mini_start}')
        mini_start = time.time()
        uc, u_diff, uv, u_next = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda()
        B, S, C, L = uc.shape
        uc, u_diff, uv, u_next = uc.reshape(B * S, 1, L), u_diff.reshape(B * S, 1, L), uv.reshape(B * S, -1, L+1), u_next.reshape(B * S, 1, L)
        # u, u_diff, uv = u[:B*S//1024//64], u_diff[:B*S//1024//64], uv[:B*S//1024//64]
        optimizer.zero_grad()
        # output = net(net.weno(u)/dt)/dt
        k1 = 0.001 / torch.pi * net.weno(net.weno(uc)/net.dt) - net.weno(uc) * uc


        loss = criterion(output, u_diff)

        loss.backward()

        # # print(f'Backward Time: {time.time() - mini_start}')
        optimizer.step()
        lr_scheduler.step()
        # print(f'Optimize Time: {time.time() - mini_start}')
        if i % 100 == 0:
            print(f'Epoch {epoch}, Batch {i}, Loss {loss.item()}, Time {time.time() - start}')
            # print(u_diff[0, 0], output[0, 0])
            # start = time.time()
            # # print(f'Output: {output[0, 0, :]}  Ground Truth: {u_diff[0, :]}')
            # plt.plot(u_next[0, 0].detach().cpu().numpy(), 'r')
            # plt.plot(output[0, 0].detach().cpu().numpy(), 'b')
            # # plt.plot(u_diff[0, :].detach().cpu().numpy(), 'k')
            # # plt.plot(output[0, 0, :].detach().cpu().numpy(), 'b')
            # plt.savefig('test.png')
            # plt.close()
        if (i+1) % 1000 == 0:
            torch.save(net.state_dict(), f'./ckpt/stencilCNN.pt')
        
    print('')

