from model import stencilCNN1D, PDE
import torch
from matplotlib import pyplot as plt
import time
from dataset import get_dataset
import yaml
import argparse
from tqdm import tqdm



parser = argparse.ArgumentParser()
# parser.add_argument('--roll_out', action='store_true')
parser.add_argument('--pde_type', type=str, default='DiffusionSorption1D')
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()

pde_type = args.pde_type

with open(f'configs/{pde_type}.yaml') as f:
    pde_config = yaml.load(f, Loader=yaml.FullLoader)

train_config = pde_config['train_config']
data_config = pde_config['data_config']
dt = train_config['dt']
dx = (train_config['x_max'] - train_config['x_min']) / train_config['num_cells']

for pde_name in data_config.keys():
    print(f'------- Evaluating {pde_name} --------------')
    # pde_type = pde_name.split('-')[0]

    # cfl_number = config['cfl_number']
    # num_cells = config['num_cells']
    # x_min = config['x_min']
    # x_max = config['x_max']
    # xv = torch.linspace(x_min, x_max, num_cells + 1)
    # xc = 0.5 * (xv[1:] + xv[:-1])
    # dx = (x_max - x_min) / num_cells


    print('------- Reading Dataset --------------')
    dataset = get_dataset(pde_type, data_config[pde_name])
    # dataset = FourierDataset(1024, 16, 1, 0.5, 10000, dt, scale=32)
    print('------- Dataset loaded --------------')
    # dataset = WenoDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=False)

    net = stencilCNN1D(dt, dx, pde_type, data_config[pde_name]['params']).cuda()
    net.load_state_dict(torch.load('./ckpt/stencilCNN.pt'))
    net.eval()
    # pde = PDE(net, pde_type, data_config[pde_name]['bc'], data_config[pde_name]['params'])
    roll_out = train_config['roll_out']
    criterion = torch.nn.MSELoss(reduction='none')

    losses = []
    mini_start = start = time.time()
    for ro in roll_out:
        print('Evaluating Roll Out:', ro)
        losses = []
        for i, data in enumerate(dataloader):
            # print(f'Data Time: {time.time() - mini_start}')
            # num_cells = 2000
            mini_start = time.time()
            u = data.cuda()
            # u = u[:, :, :4:]
            # B, T, L = u.shape
            # xv = torch.linspace(-1, 1, num_cells + 1, dtype=torch.float64)
            # xc = 0.5 * (xv[1:] + xv[:-1])
            # u = xc**2
            # u = u.unsqueeze(0).unsqueeze(0).cuda()
            
            # for j in tqdm(range(T-ro)):
            j=0
            uc = u[:, j:j+1]
            for k in range(train_config['record']*ro):
            # for k in range(config['record']):
                with torch.no_grad():
                    du = net(uc)
                    uc = uc + du
            # print(du[1])
            # print(u[1, j])
            loss = torch.sqrt(criterion(uc[:, 0], u[:, j+ro])).mean()
            losses.append(loss.item())

            if i%1 == 0:
                print(f'Iter {i}, Loss: {loss.item()}')
                error = uc[:, 0] - u[:, j+ro]
                # plt.plot(error[1, :].detach().cpu().numpy())
                # plt.savefig(f'./figs/erro_{k}_{j}_{i}.png')
                # plt.close()

                if args.plot:
                    plt.plot(u[1, j+ro, :].detach().cpu().numpy(), 'r', label='gt')
                    plt.plot(uc[1, 0, :].detach().cpu().numpy(), 'b', label='next')
                    plt.plot(error[1, :].detach().cpu().numpy(), 'g', label='error')
                    plt.legend()
                    plt.savefig(f'./figs/pred_{k}_{j}_{i}.png')
                    plt.close()

        losses = torch.tensor(losses).mean()
        print(f'Final Mean Loss for {pde_name} with rollout = {ro} is: {losses}')
        print('------------------------------------\n\n')
        # output = torch.stack(outputs, dim=1).squeeze(2)
        
        # loss = torch.sqrt(criterion(output, u[:, 1:])).mean(dim=(1, 2))

        # if torch.isnan(loss).any():
        #     nan_mask = torch.isnan(output).sum(dim=(1, 2))
        #     valid_indices = torch.nonzero(nan_mask == 0).squeeze()
        #     print(f"Nan Loss at {i}, Valid percentage is {valid_indices.numel() / config['batch_size']}")

        #     if valid_indices.numel() > 0:
        #         loss = loss[valid_indices].mean()
        #         losses.append(loss.item())
        #     else:
        #         print(f'All losses are nan at {i}')
        #         continue
        # else:
        #     losses.append(loss.mean().item())

    # if len(losses) == 0:
    #     print('All losses are nan')
    # else:
    #     losses = torch.tensor(losses)






