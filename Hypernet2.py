import torch
import torch.nn as nn
import numpy as np
from models import MODEL1
import torchaudio

def log_to_norm(log_val):
    norm_val = (np.log10(log_val) - np.log10(1000)) / (np.log10(500000) - np.log10(1000)) #pot range 1k - 500k
    return norm_val

def lin_to_norm(lin_val):
    norm_val = (lin_val - 1000) / (19000) #lin pot range 1k-20k
    return norm_val



def normalize(controls):
    # Apply lin_to_norm on all elements in the first column (tone control)
    tone_norm = lin_to_norm(controls[:, 0])

    # Apply log_to_norm on all elements in the second column (gain control)
    gain_norm = log_to_norm(controls[:, 1])

    # Stack the normalized values along dimension 1 to get a shape of (N, 2)
    normalized_controls = torch.stack([tone_norm.float(), gain_norm.float()], dim=1)

    return normalized_controls  # Move to the original device (if needed)
def assign_params(hyper, endpoint):
    with torch.no_grad():
        endpoint.mlp1[0].weight.data = hyper['mlp1_0'][0]
        endpoint.mlp1[0].bias.data = hyper['mlp1_0'][1]

        endpoint.mlp1[2].weight.data  = hyper['mlp1_2'][0]
        endpoint.mlp1[2].bias.data  = hyper['mlp1_2'][1]

        endpoint.mlp1[4].weight.data  = hyper['mlp1_4'][0]
        endpoint.mlp1[4].bias.data  = hyper['mlp1_4'][1]

        endpoint.mlp1[6].weight.data  = hyper['mlp1_6'][0]
        endpoint.mlp1[6].bias.data  = hyper['mlp1_6'][1]

        for i in range(len(endpoint.filters)):
            endpoint.filters[i].g.data  = hyper['g'][i]
            endpoint.filters[i].R.data  = hyper['R'][i]
            endpoint.filters[i].m_hp.data  = hyper['m_hp'][i]
            endpoint.filters[i].m_bp.data  = hyper['m_bp'][i]
            endpoint.filters[i].m_lp.data  = hyper['m_lp'][i]

        endpoint.mlp2[0].weight.data  = hyper['mlp2_0'][0]
        endpoint.mlp2[0].bias.data  = hyper['mlp2_0'][1]

        endpoint.mlp2[2].weight.data  = hyper['mlp2_2'][0]
        endpoint.mlp2[2].bias.data  = hyper['mlp2_2'][1]

        endpoint.mlp2[4].weight.data  = hyper['mlp2_4'][0]
        endpoint.mlp2[4].bias.data  = hyper['mlp2_4'][1]

        endpoint.mlp2[6].weight.data  = hyper['mlp2_6'][0]
        endpoint.mlp2[6].bias.data  = hyper['mlp2_6'][1]

class HyperNetwork(nn.Module):
    def __init__(self):
        super(HyperNetwork, self).__init__()

        # For simplicity, the hypernetwork will have one shared latent space
        # and split branches for each parameter set in the target model.
        latent_out = 4
        # This layer processes the 2-dimensional input to a higher-dimensional latent space
        self.fc1 = nn.Linear(2, latent_out)  # Latent space mapping
        #non linearity needed? filter params and biases have greater abs values than -1 and 1
        #self.fc2 = nn.ReLU()

        # Each of these layers will generate the corresponding weights and biases.

        # MLP1 weights and biases
        self.mlp1_0_weight = nn.Linear(latent_out, 6 * 1)  # Generates [6, 1] weight
        self.mlp1_0_bias = nn.Linear(latent_out, 6)  # Generates [6] bias
        self.mlp1_2_weight = nn.Linear(latent_out, 8 * 3)  # Generates [8, 3] weight
        self.mlp1_2_bias = nn.Linear(latent_out, 8)  # Generates [8] bias
        self.mlp1_4_weight = nn.Linear(latent_out, 10 * 4)  # Generates [10, 4] weight
        self.mlp1_4_bias = nn.Linear(latent_out, 10)  # Generates [10] bias
        self.mlp1_6_weight = nn.Linear(latent_out, 5 * 5)  # Generates [5, 5] weight
        self.mlp1_6_bias = nn.Linear(latent_out, 5)  # Generates [5] bias

        # Filters parameters (1D scalar values for each filter)
        self.filters_g = nn.Linear(latent_out, 5)  # Generates [5] values for g
        self.filters_R = nn.Linear(latent_out, 5)  # Generates [5] values for R
        self.filters_m_hp = nn.Linear(latent_out, 5)  # Generates [5] values for m_hp
        self.filters_m_bp = nn.Linear(latent_out, 5)  # Generates [5] values for m_bp
        self.filters_m_lp = nn.Linear(latent_out, 5)  # Generates [5] values for m_lp

        # MLP2 weights and biases
        self.mlp2_0_weight = nn.Linear(latent_out, 10 * 5)  # Generates [10, 5] weight
        self.mlp2_0_bias = nn.Linear(latent_out, 10)  # Generates [10] bias
        self.mlp2_2_weight = nn.Linear(latent_out, 8 * 5)  # Generates [8, 5] weight
        self.mlp2_2_bias = nn.Linear(latent_out, 8)  # Generates [8] bias
        self.mlp2_4_weight = nn.Linear(latent_out, 6 * 4)  # Generates [6, 4] weight
        self.mlp2_4_bias = nn.Linear(latent_out, 6)  # Generates [6] bias
        self.mlp2_6_weight = nn.Linear(latent_out, 1 * 3)  # Generates [1, 3] weight
        self.mlp2_6_bias = nn.Linear(latent_out, 1)  # Generates [1] bias



    def forward(self, z, x):

        # z is a 2-dimensional input vector - normalized 0-1 user inputs
        #print(z)
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = torch.device('cpu')
        # Normalize and pass through the latent space
        z = normalize(z)
        z = self.fc1(z)

        batch_size = z.shape[0]

        # Generate weights and biases for each layer with batch dimension
        w_mlp1_0 = self.mlp1_0_weight(z).view(batch_size, 6, 1)
        b_mlp1_0 = self.mlp1_0_bias(z).view(batch_size, 6)

        w_mlp1_2 = self.mlp1_2_weight(z).view(batch_size, 8, 3)
        b_mlp1_2 = self.mlp1_2_bias(z).view(batch_size, 8)

        w_mlp1_4 = self.mlp1_4_weight(z).view(batch_size, 10, 4)
        b_mlp1_4 = self.mlp1_4_bias(z).view(batch_size, 10)

        w_mlp1_6 = self.mlp1_6_weight(z).view(batch_size, 5, 5)
        b_mlp1_6 = self.mlp1_6_bias(z).view(batch_size, 5)

        # Generate scalar parameters for filters with batch dimension
        g = self.filters_g(z).view(5, batch_size)
        print(g)
        print(g.shape)
        R = self.filters_R(z).view(5, batch_size)
        m_hp = self.filters_m_hp(z).view(5, batch_size)
        m_bp = self.filters_m_bp(z).view(5, batch_size)
        m_lp = self.filters_m_lp(z).view(5, batch_size)

        w_mlp2_0 = self.mlp2_0_weight(z).view(batch_size, 10, 5)
        b_mlp2_0 = self.mlp2_0_bias(z).view(batch_size, 10)

        w_mlp2_2 = self.mlp2_2_weight(z).view(batch_size, 8, 5)
        b_mlp2_2 = self.mlp2_2_bias(z).view(batch_size, 8)

        w_mlp2_4 = self.mlp2_4_weight(z).view(batch_size, 6, 4)
        b_mlp2_4 = self.mlp2_4_bias(z).view(batch_size, 6)

        w_mlp2_6 = self.mlp2_6_weight(z).view(batch_size, 1, 3)
        b_mlp2_6 = self.mlp2_6_bias(z).view(batch_size, 1)

        dummy_model = MODEL1([3,4,5], 5, 32768)

        #assign parameters to dummy model
        with torch.no_grad():
            dummy_model.mlp1[0].weight.data = w_mlp1_0
            dummy_model.mlp1[0].bias.data = b_mlp1_0

            dummy_model.mlp1[2].weight.data = w_mlp1_2
            dummy_model.mlp1[2].bias.data = b_mlp1_2

            dummy_model.mlp1[4].weight.data = w_mlp1_4
            dummy_model.mlp1[4].bias.data = b_mlp1_4

            dummy_model.mlp1[6].weight.data = w_mlp1_6
            dummy_model.mlp1[6].bias.data = b_mlp1_6

            for i in range(len(dummy_model.filters)):
                dummy_model.filters[i].g.data = g[i]
                dummy_model.filters[i].R.data = R[i]
                dummy_model.filters[i].m_hp.data = m_hp[i]
                dummy_model.filters[i].m_bp.data = m_bp[i]
                dummy_model.filters[i].m_lp.data = m_lp[i]

            print(dummy_model.filters[0].g)

            dummy_model.mlp2[0].weight.data = w_mlp2_0
            dummy_model.mlp2[0].bias.data = b_mlp2_0

            dummy_model.mlp2[2].weight.data = w_mlp2_2
            dummy_model.mlp2[2].bias.data = b_mlp2_2

            dummy_model.mlp2[4].weight.data = w_mlp2_4
            dummy_model.mlp2[4].bias.data = b_mlp2_4

            dummy_model.mlp2[6].weight.data = w_mlp2_6
            dummy_model.mlp2[6].bias.data = b_mlp2_6

            #generate prediction from dummy model
            y_hat = dummy_model(x).squeeze(-1)

        # Return all generated weights and biases
        return y_hat

if __name__ == '__main__':
    # Example usage:
    # Initialize hypernetwork
    hypernetwork = HyperNetwork()


    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    # Create a random latent vector of size 2
    # latent_vector = torch.randn(1, 2)
    # latent_vector = latent_vector / max(latent_vector)
    latent_vector = torch.tensor([[1000, 1000]])
    #latent_vector = torch.tensor([[20000, 500000],[15000, 100000],[10000, 1000],[20000, 20000]])


    dir = 'ts808_bigtrain_MODEL1'

    params = []
    file = open('results/' + dir + '/parameters.txt', 'r')
    for i, line in enumerate(file.readlines()):
         if i <= 5:
            tmp = line.split()
            if i == 0:
                data = tmp[-1]
            else:
                params.append(tmp[-1])
    file.close()
    print('Model: ' + dir)

    layers = [int(i) for i in params[1].strip("[]").split(",")]
    layer = int(params[2])
    n = int(params[3])
    N = int(params[4])

    train_input, fs = torchaudio.load(r'D:\fau\magisterka\spice\out\train\\ts808-input.wav')
    print((train_input.view(1,-1,1)).shape)
    hypernetwork(latent_vector, train_input)



    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Shape: {param.shape}")
    #     print('Data: ', param.data)
    # print(model.mlp1[0].weight.data)
    # print(generated_params['mlp1_0'][0])

