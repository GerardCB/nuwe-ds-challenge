import torch
import torch.nn as nn

# ----- Baseline -----
class BaselineNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        """
        Args:
            input_size: the size of the input
            hidden_size: the size of the hidden layers
            num_layers: the number of hidden layers
            dropout: the dropout rate
        """
        super(BaselineNet, self).__init__()
        self.fc = nn.Sequential()  # Fully connected layers

        # First layer
        self.fc.add_module('fc0', nn.Linear(input_size, hidden_size))
        self.fc.add_module('relu0', nn.ReLU())
        self.fc.add_module('dropout0', nn.Dropout(dropout))

        # Hidden layers
        for i in range(1, num_layers):
            self.fc.add_module(f'fc{i}', nn.Linear(hidden_size, hidden_size))
            self.fc.add_module(f'relu{i}', nn.ReLU())
            self.fc.add_module(f'dropout{i}', nn.Dropout(dropout))
            
        # Last layer
        self.fc.add_module(f'fc{num_layers}', nn.Linear(hidden_size, 1))
        self.fc.add_module(f'sigmoid', nn.Sigmoid())

    def forward(self, x):
        x = self.fc(x)
        x = x.squeeze(1)
        return x
    
# ----- FT Transformer -----
class NumericalEmbedder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NumericalEmbedder, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gates)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.2):
        super().__init__()
        self.head_dim = dim // heads
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        qkv = self.to_qkv(self.norm(x)).view(x.shape[0], -1, self.heads, 3, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=-3)
        out = torch.einsum('hbid,hbjd->bhij', q, k) / self.head_dim ** 0.5
        out = torch.einsum('bhij,hbjd->bhid', out.softmax(dim=-1), v)
        out = self.to_out(out.transpose(1, 2).contiguous().view(x.shape)) + x
        return self.dropout(out)
    
class FTTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, d_ff, dropout=0.2):
        """
         Args:
            input_dim: the dimension of the input
            d_model: the latent dimension of the embeddings
            nhead: number of attention heads
            num_layers: the number of hidden layers
            d_ff: hidden dimension of the feed forward layer
            dropout: the dropout rate
        """
        super().__init__()
        self.numerical_embedder = NumericalEmbedder(input_dim, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(d_model, nhead, dropout),
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_ff),
                    GEGLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff // 2, d_model)
                )
            ]) for _ in range(num_layers)
        ])
        self.to_logits = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.numerical_embedder(x)
        for layer in self.layers:
            x = layer[0](x) + x
            x = layer[1](x) + x
        return self.to_logits(x).squeeze(1)
