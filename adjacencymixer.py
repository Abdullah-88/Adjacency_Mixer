import torch
from torch import nn

class VecDyT(nn.Module):
    def __init__(self, input_shape):

        super().__init__()

        self.alpha = nn.Parameter(torch.randn(input_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x

class VecDyGeluSine(nn.Module):
    def __init__(self, input_shape):

        super().__init__()

        self.alpha = nn.Parameter(torch.randn(input_shape))
        self.beta = nn.Parameter(torch.randn(input_shape))
        self.gamma = nn.Parameter(torch.randn(1))
        self.etta = nn.Parameter(torch.randn(1))
        self.gelu = nn.GELU()

    def forward(self, x):

        x = self.gamma * self.gelu(self.alpha * x) + self.etta * torch.sin(self.beta * x)

        return x

class GatedProjection(nn.Module):
    def __init__(self,dim):

        super().__init__()

        self.proj =  nn.Linear(dim, dim, bias=False)
        self.modulate = VecDyGeluSine(dim)

    def forward(self, x):

        u, v = x, x

        u = self.modulate(u)
        v = self.proj(v)
        g = u * v

        return g

class Mixer(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        
        self.src = GatedProjection(in_features)
        self.dst = GatedProjection(in_features)
        
    def forward(self, x, temperature=0.2):
       
        src = self.src(x)
        dst = self.dst(x)
       
        scores = torch.matmul(src, dst.transpose(1, 2)) / (src.size(-1) ** 0.5)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-20) + 1e-20)
        adj = torch.sigmoid((scores + gumbel_noise) / temperature)
        
        aggregated = torch.matmul(adj, dst)
        
        return aggregated

class AdjacencyMixerBlock(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mixer = Mixer(dim)
        self.norm1 = VecDyT(dim)
        self.norm2 = VecDyT(dim)
        self.ff = GatedProjection(dim)
                
    def forward(self, x):
    
        residual = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + residual
        
        return x

class AdjacencyMixer(nn.Module):
   
    def __init__(self, d_model, num_layers):
        super().__init__()
        self.model = nn.Sequential(
            *[AdjacencyMixerBlock(d_model) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.model(x)