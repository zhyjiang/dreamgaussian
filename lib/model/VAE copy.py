
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid


import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class GSData(Dataset):
    def __init__(self, path):
        self.path = path
        # self.type = ty
        # import ipdb; ipdb.set_trace()
        self.data_scale = np.load(self.path,allow_pickle=True)['scales']
        self.data_rots = np.load(self.path,allow_pickle=True)['rots']
        # self.data = np.concatenate((self.data_scale,self.data_rots),axis=-1)

    def __len__(self):
        return len(self.data_scale)

    def __getitem__(self, idx):
        scale = self.data_scale[idx]
        rot = self.data_rots[idx]
        sample = np.concatenate((scale,rot),axis=-1)
        return sample

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5,opt=None):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self.opt  = opt
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        # import ipdb;ipdb.set_trace()

        # if len(inputs.shape)!=4:
        #     inputs = inputs.unsqueeze(0)

        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        # import ipdb;ipdb.set_trace()

        input_shape = inputs.shape

        # import ipdb;ipdb.set_trace()

        

        # Flatten input
        flat_input = inputs.reshape(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)


        

        # import ipdb;ipdb.set_trace()
        # Use EMA to update the embedding vectors
        # if self.training and self.opt.vae_tune:
        #     self._ema_cluster_size = self._ema_cluster_size * self._decay + \
        #                              (1 - self._decay) * torch.sum(encodings, 0)
            
        #     # Laplace smoothing of the cluster size
        #     n = torch.sum(self._ema_cluster_size.data)
        #     self._ema_cluster_size = (
        #         (self._ema_cluster_size + self._epsilon)
        #         / (n + self._num_embeddings * self._epsilon) * n)
            
        #     dw = torch.matmul(encodings.t(), flat_input)
        #     self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
        #     self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        

        # if self.opt.vae_tune:
        # # Loss
        #     e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        #     loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        # avg_probs = torch.mean(encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        import ipdb; ipdb;ipdb.set_trace()
        
        # convert quantized from BHWC -> BCHW
        return None, quantized.permute(0, 3, 1, 2).contiguous(), None , encodings
    
    
    
    
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                    out_channels=num_residual_hiddens,
                    kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                    out_channels=num_hiddens,
                    kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
    
    
    
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=num_hiddens//2,
                                kernel_size=3,
                                stride=1, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                out_channels=num_hiddens,
                                kernel_size=3,
                                stride=1, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                out_channels=num_hiddens,
                                kernel_size=3,
                                stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                            num_hiddens=num_hiddens,
                                            num_residual_layers=num_residual_layers,
                                            num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        # import ipdb; ipdb.set_trace()
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)
    
    
class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=num_hiddens,
                                kernel_size=3, 
                                stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                            num_hiddens=num_hiddens,
                                            num_residual_layers=num_residual_layers,
                                            num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=3, 
                                                stride=1, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=7,
                                                kernel_size=3, 
                                                stride=1, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)
    
    
class Model(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()
        
        self._encoder = Encoder(7, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)

    def forward(self, x):
        x = x.unsqueeze(1).permute(0, 3, 1, 2)
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        _, quantized, _, _ = self._vq_vae(z)
        import ipdb;ipdb.set_trace()
        x_recon = self._decoder(quantized)
        return loss, x_recon, _


def compute_data_variance(data_loader, model, device):
    data_variance = torch.zeros(7).to(device)
    for i, data in enumerate(data_loader):
        data = data.to(device).float()
        # import ipdb; ipdb.set_trace()
        data_variance += torch.var(data, 1).squeeze()
    data_variance /= len(data_loader)
    return data_variance


batch_size = 1
num_training_updates = 15000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 128
num_embeddings = 4096*2

commitment_cost = 0.25

decay = 0.99

learning_rate = 5e-4

import torch
from torch.utils.tensorboard import SummaryWriter

output_root = 'vae_model_gs'
os.makedirs(output_root, exist_ok=True)

# Create a SummaryWriter instance
writer = SummaryWriter(os.path.join(output_root,'logs'))



model = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim, 
              commitment_cost, decay).cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
model.load_state_dict(torch.load('/home/zhongyuj/projects/GauHuman/vae_model_gs/model_1100.pth'))
model.eval()
train_res_recon_error = []
train_res_perplexity = []


training_data = GSData('vae_gs.npz')

training_loader = DataLoader(training_data, 
                             batch_size=batch_size, 
                             shuffle=True,
                             pin_memory=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data_variance = torch.tensor(np.var(training_data.data,axis=0)).cuda().float()
data_variance = compute_data_variance(training_loader, model, device)
# import ipdb; ipdb.set_trace()

with torch.no_grad():
    for i in tqdm(range(3000)):
        for _,data in enumerate(training_loader):
            data = torch.tensor(data).to(device).float()
            optimizer.zero_grad()

            vq_loss, data_recon, perplexity = model(data)
            # import ipdb; ipdb.set_trace()
            data_recon = data_recon.squeeze(2).permute(0, 2, 1)
            # data_variance = torch.var(training_data.data,axis=0).cuda().float()
            recon_error = F.mse_loss(data_recon, data) / data_variance
            loss = recon_error + vq_loss
            loss = loss.mean()
            # import ipdb; ipdb.set_trace()
            writer.add_scalar('training_loss', loss.item(), i)
            writer.add_scalar('recon_loss', torch.mean(recon_error).item(), i)

            writer.add_scalar('vq_loss', vq_loss.item(), i)

            loss.backward()
            optimizer.step()

            train_res_recon_error.append(recon_error.mean().item())
            train_res_perplexity.append(perplexity.mean().item())

            if (i+1) % 100 == 0:
                torch.save(model.state_dict(), os.path.join(output_root,'model_%d.pth' % (i+1)))

            # if (i+1) % 4 == 0:
            writer.add_scalar('recon_error_smooth', np.mean(train_res_recon_error[-100:]), i)
            writer.add_scalar('perplexity_smooth', np.mean(train_res_perplexity[-100:]), i)




writer.close()