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

dataset = FourierDataset(num_cells=num_cells, series=16, wavespeed=1, cfl_number=0.5, num_samples=81920, scale=1024)
# dataset = WenoDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

net = stencilCNN(dt, dx).cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.train_params(), lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 45 * 100 )

for epoch in range(10000):
    mini_start = start = time.time()
    for i, data in enumerate(dataloader):
        # print(f'Data Time: {time.time() - mini_start}')
        mini_start = time.time()
        u, u_diff, uv = data[0].cuda(), data[1].cuda(), data[2].cuda()
        B, S, C, L = u.shape
        u, u_diff, uv = u.reshape(B * S, 1, L), u_diff.reshape(B * S, 1, L), uv.reshape(B * S, -1, L+1)
        # u, u_diff, uv = u[:B*S//1024//64], u_diff[:B*S//1024//64], uv[:B*S//1024//64]
        optimizer.zero_grad()
        output = net(u)

        if i % 5 == 0:
            print(net.stencil.data)
            print(net.stencil.data.sum().item())
        # print(f'Forward Time: {time.time() - mini_start}')
        loss = criterion(-output/dt, u_diff)
        # torch.save(output, f'dataset/uv{i}.pt')
        # torch.save(u, f'dataset/u{i}.pt')
        loss.backward()

        # # print(f'Backward Time: {time.time() - mini_start}')
        optimizer.step()
        lr_scheduler.step()
        # print(f'Optimize Time: {time.time() - mini_start}')
        if i % 5 == 0:
            print(f'Epoch {epoch}, Batch {i}, Loss {loss.item()}, Time {time.time() - start}')
            # start = time.time()
            # # print(f'Output: {output[0, 0, :]}  Ground Truth: {u_diff[0, :]}')
            # plt.plot(uv[0, 0].detach().cpu().numpy(), 'r')
            # # plt.plot(u_diff[0, :].detach().cpu().numpy(), 'k')
            # # plt.plot(output[0, 0, :].detach().cpu().numpy(), 'b')
            # plt.savefig('test.png')
            # plt.close()

        
    print('')

