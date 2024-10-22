import torch
import torch.nn as nn
import numpy as np
from models import MODEL1

def log_to_norm(log_val):
    norm_val = (np.log10(log_val) - np.log10(1000)) / (np.log10(500000) - np.log10(1000)) #pot range 1k - 500k
    return norm_val

def lin_to_norm(lin_val):
    norm_val = (lin_val - 1000) / (19000) #lin pot range 1k-20k
    return norm_val

def normalize(controls):
    gain_norm = log_to_norm(controls[1].cpu())
    tone_norm = lin_to_norm(controls[0].cpu())
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor([tone_norm, gain_norm], dtype = torch.float32)

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



    def forward(self, z):

        # z is a 2-dimensional input vector - normalized 0-1 user inputs
        #print(z)
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device('cpu')
        z = normalize(z)#.to(device)
        #print(z)
        z = self.fc1(z)  # Process through the latent space


        # Generate weights and biases for each layer
        w_mlp1_0 = self.mlp1_0_weight(z).view(6, 1)
        b_mlp1_0 = self.mlp1_0_bias(z).squeeze(0)

        w_mlp1_2 = self.mlp1_2_weight(z).view(8, 3)
        b_mlp1_2 = self.mlp1_2_bias(z).squeeze(0)

        w_mlp1_4 = self.mlp1_4_weight(z).view(10, 4)
        b_mlp1_4 = self.mlp1_4_bias(z).squeeze(0)

        w_mlp1_6 = self.mlp1_6_weight(z).view(5, 5)
        b_mlp1_6 = self.mlp1_6_bias(z).squeeze(0)

        # Generate scalar parameters for filters
        g = self.filters_g(z)
        R = self.filters_R(z)
        m_hp = self.filters_m_hp(z)
        m_bp = self.filters_m_bp(z)
        m_lp = self.filters_m_lp(z)

        w_mlp2_0 = self.mlp2_0_weight(z).view(10, 5)
        b_mlp2_0 = self.mlp2_0_bias(z).squeeze(0)

        w_mlp2_2 = self.mlp2_2_weight(z).view(8, 5)
        b_mlp2_2 = self.mlp2_2_bias(z).squeeze(0)

        w_mlp2_4 = self.mlp2_4_weight(z).view(6, 4)
        b_mlp2_4 = self.mlp2_4_bias(z).squeeze(0)

        w_mlp2_6 = self.mlp2_6_weight(z).view(1, 3)
        b_mlp2_6 = self.mlp2_6_bias(z).squeeze(0)

        # Return all generated weights and biases
        return {
            "mlp1_0": (w_mlp1_0, b_mlp1_0),
            "mlp1_2": (w_mlp1_2, b_mlp1_2),
            "mlp1_4": (w_mlp1_4, b_mlp1_4),
            "mlp1_6": (w_mlp1_6, b_mlp1_6),

            #"filters": (g, R, m_hp, m_bp, m_lp),

            'g' : g,
            'R' : R,
            'm_hp' : m_hp,
            'm_bp' : m_bp,
            'm_lp' : m_lp,
            "mlp2_0": (w_mlp2_0, b_mlp2_0),
            "mlp2_2": (w_mlp2_2, b_mlp2_2),
            "mlp2_4": (w_mlp2_4, b_mlp2_4),
            "mlp2_6": (w_mlp2_6, b_mlp2_6)
        }

if __name__ == '__main__':
    # Example usage:
    # Initialize hypernetwork
    hypernetwork = HyperNetwork().cuda()


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
    latent_vector = torch.tensor([1000, 1000])

    # Generate the weights and biases
    generated_params = hypernetwork(latent_vector)
    #print(get_n_params(hypernetwork))
    # Print generated parameters for one of the layers (for example mlp1_0)
    # print(generated_params.keys())
    for key in generated_params.keys():
        for i in generated_params[key]:
            print("Layer: ", key, "Shape: ", i.shape)
            #print(generated_params[key][i])
    #print(f"Generated mlp1_0 weight: {generated_params['mlp1_0'][0].shape}")
    #print(f"Generated mlp1_0 bias: {generated_params['mlp1_0'][1].squeeze(0).shape}")

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

    model = MODEL1(layers, n, N)
    pth = fr'D:\fau\magisterka\KLANN\results\{dir}\model.pth'

    state_dict = torch.load(pth)
    #model.load_state_dict(state_dict)
    assign_params(generated_params, model)
    print(hypernetwork.parameters()[1])

    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Shape: {param.shape}")
    #     print('Data: ', param.data)
    # print(model.mlp1[0].weight.data)
    # print(generated_params['mlp1_0'][0])

