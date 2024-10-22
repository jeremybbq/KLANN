import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from models import MODEL1
from Hypernet import HyperNetwork, assign_params
from hyper_dataset import load_data
import time

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Directory "{directory}" created.')
    else:
        print(f'Directory "{directory}" already exists.')

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("using", device)

dir = './hyper'
create_directory_if_not_exists(dir)

# GLU MLP hidden layer sizes
layers = [3,4,5]
# FC layer size in MODEL2
layer = 5
# number of biquads
n = 5
# length for estimated FIR filter length
N = 32768
# samples used for calculating the loss
seq_length = 1024

n_epochs = 1000
# samples used for dividing the audio
# (seq_length and trunc_length should sum to a multiple of N)
# (1*N -> no overlap-add method)
batch_size = 1
learning_rate = 1e-3
loss_func = nn.MSELoss()
alpha = 0.001

hyper_model = HyperNetwork().train(True)
dummy_model = MODEL1(layers, n, N)

train_dataset = load_data(r'D:\fau\magisterka\KLANN\results\train_set', 'ts808', batch_size)
hyper_model_optimizer = torch.optim.Adam(hyper_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


def train_loop():
    train_loss = 0
    for(ctr, pretrained_params) in train_dataset:
        ctr = ctr.squeeze(0)
        ctr_in = ctr
        hyper_model_optimizer.zero_grad()

        #get params from prediction by HN
        hyper_out = hyper_model(ctr_in)
        assign_params(hyper_out, dummy_model)
        gen_params = []
        for name, param in dummy_model.named_parameters():

            gen_params.append(param)

        for i in range(len(pretrained_params)):
            if i == 0:
                loss = loss_func(gen_params[i], pretrained_params[i].squeeze(0))
            else:

                loss += loss_func(gen_params[i], pretrained_params[i].squeeze(0))


        #loss /= len(pretrained_params_out)

        loss.backward()
        print(loss.grad)
        hyper_model_optimizer.step()

        print(list(hyper_model.parameters())[0].grad)
        # for i, param in enumerate(hyper_model.parameters()):
        #     if i ==1:
        #         print(param)

        train_loss += loss.item()

    return train_loss / len(train_dataset)

begin = time.time()
best_loss = float('inf')
for epoch in range(n_epochs):
    hyper_model.train(True)
    train_loss = train_loop()
    if train_loss < best_loss:
        best_loss = train_loss
        print('new best loss:')
        print(f'Epoch: {epoch + 1}; train loss = ', train_loss)
        torch.save(hyper_model.state_dict(), dir + '/model.pth')
        torch.save(hyper_model_optimizer.state_dict(), dir + '/model_optimizer.pth')
    if epoch % 50 == 0:
        print(f'Epoch: {epoch + 1}; train loss = ', train_loss)


print("Time elapsed: ", (time.time() - begin) / 60, " mins")