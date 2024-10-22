import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import MODEL1
import os



def count_directories(path):
    return sum(os.path.isdir(os.path.join(path, entry)) for entry in os.listdir(path))

def load_data(target, effect, batch_size):
    data = HyperSet(target, effect)
    return DataLoader(dataset=data, batch_size=batch_size, shuffle=True)

class HyperSet(Dataset):
    def __init__(self, target_dir, data):
        self.target_dir = target_dir
        self.data = data

        layers = [3,4,5]
        n = 5
        N = 32768

        self.dummy_model = MODEL1(layers, n, N)

        self.Rt = ['1k', '5k', '10k', '15k', '20k']
        self.Rg = ['1k', '5k', '20k', '100k', '500k']

    def __getitem__(self, index): #index is int :v

        #fix when whole data
        ctr = [0, 0]
        ctrRt = [1000, 5000, 10000, 15000, 20000]
        ctrRg = [1000, 5000, 20000, 100000, 500000]
        if index<=4:
            Rt_ctrl = self.Rt[index]
            ctr[0] = ctrRt[index]
            Rg_ctrl = '1k'
            ctr[1] = 1000
        elif index > 4:
            Rt_ctrl = '1k'
            ctr[0] = 1000
            Rg_ctrl = self.Rg[index - 4]
            ctr[1] = ctrRg[index - 4]

        dir = os.path.join(self.target_dir,self.data + fr'_Rt_{Rt_ctrl}_Rg_{Rg_ctrl}\model.pth' )
        state_dict = torch.load(dir)
        self.dummy_model.load_state_dict(state_dict)

        param_arr = []

        for param in self.dummy_model.parameters():
            if param.shape == torch.Size([1]):
                param_arr.append(param.squeeze(0))
            else:
                param_arr.append(param.data)

        return torch.tensor(ctr), param_arr

    def __len__(self):
        return count_directories(self.target_dir)

if __name__ == '__main__':
    dataload = load_data(r'D:\fau\magisterka\KLANN\results\train_set', 'ts808', 1)
    for i_batch, sample in enumerate(dataload):
        if i_batch ==3:
            ctr, model = sample
            print(i_batch)
            print(ctr.squeeze(0))
            #print(len(model))
            for param in model:
                param = param.squeeze(0)

                print(param.shape)
