import torch
import numpy as np
import os

import torchaudio
from torch.utils.data import Dataset, DataLoader

# create dataset
def PreProcess(train_input, train_target, sequence_length, truncate_length, batch_size):
    data = AudioDataSet(train_input, train_target, sequence_length, truncate_length)
    return DataLoader(dataset = data, batch_size = batch_size, shuffle = True)


class AudioDataSet(Dataset):
    def __init__(self, input, target_dir, sequence_length, truncate_length):
        self.Rt = ['1k', '5k', '10k', '15k', '20k']
        self.Rg = ['1k', '5k', '20k', '100k', '500k']
        self.input_sequence = self.wrap_to_sequences(input, sequence_length, truncate_length)
        self.target_sequence = self.wrap_to_sequences_target(target_dir, sequence_length, truncate_length)
        self.length = self.input_sequence.shape[0] * 25


    def __getitem__(self, index):
        conf_index = int(index // self.input_sequence.shape[0])
        seq_index = index % self.input_sequence.shape[0]
        ctr = [0, 0]
        ctrRt = [1000, 5000, 10000, 15000, 20000]
        ctrRg = [1000, 5000, 20000, 100000, 500000]
        ctr[0] = ctrRt[int(conf_index//5)]
        ctr[1] = ctrRg[int(conf_index % 5)]
        # print(ctr)
        # print(self.target_sequence[conf_index][1])
        return torch.tensor(ctr), self.input_sequence[seq_index,:,:], self.target_sequence[conf_index][seq_index,:,:]

    def __len__(self):
        return self.length

    def wrap_to_sequences(self, waveform, sequence_length, truncate_length):
        num_sequences = int(np.floor((waveform.shape[1] - truncate_length) / sequence_length))
        tensors = []
        for i in range(num_sequences):
            low = i * sequence_length
            high = low + sequence_length + truncate_length
            tensors.append(waveform[0,low:high])
        return torch.unsqueeze(torch.stack(tensors, dim = 0), dim = -1)

    def wrap_to_sequences_target(self, dir, sequence_length, truncate_length):
        sequences = []
        for Rtone in self.Rt:
            for Rgain in self.Rg:
                # print(Rtone)
                # print(Rgain)
                file = os.path.join(dir,fr'ts808-target_Rt_{Rtone}_Rg_{Rgain}.wav')
                waveform, _ = torchaudio.load(file)
                tensors = self.wrap_to_sequences(waveform, sequence_length, truncate_length)
                sequences.append(tensors)
        return sequences

if __name__ == '__main__':
    test_dict = {
        '1k' : 1000,
        '5k' : 5000,
        '10k' : 10000,
        '15k' : 15000,
        '20k' : 20000,
        '100k' : 100000,
        '500k' : 500000
    }
    N = 32768
    # samples used for calculating the loss
    seq_length = 22050
    # samples used for dividing the audio
    # (seq_length and trunc_length should sum to a multiple of N)
    # (1*N -> no overlap-add method)
    trunc_length = 1 * N - seq_length
    inp, fs = torchaudio.load(r'D:\fau\magisterka\spice\out\train\ts808-input.wav')
    data = PreProcess(inp.float(), r'D:\fau\magisterka\spice\out\train', seq_length, trunc_length, 5)

    for batch, sample in enumerate(data):
        ctr, input, output = sample
        if batch == 3:
            print(ctr)
            print(input.shape)
            print(output.shape)
        # if test_dict[output[0][0]] != ctr.squeeze(0)[0].item():
        #     print(output[0][0])
        #     print(ctr.squeeze(0)[0].item())
        # if test_dict[output[1][0]] != ctr.squeeze(0)[1].item():
        #     print(output[1][0])
        #     print(ctr.squeeze(0)[1].item())

