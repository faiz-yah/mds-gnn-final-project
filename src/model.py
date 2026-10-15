class DenseGCNLayer(nn.Module):
    """One GCN layer, dense form: A_hat @ X @ W + b (bias added after aggregation).

    Expects a pre-normalised adjacency (see data.normalize_adjacency). A_hat
    broadcasts over any leading dimensions, so x may be [N, F], [B, N, F] or
    [B, T, N, F] — the same layer serves every timestep of every sample at once.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        return a_hat @ self.linear(x) + self.bias


class STGNN(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_dim: int,
                 num_gcn_layers:int,
                 horizon:int,
                 dropout:float
                )->None:
        super().__init__()
        
        dims = [in_features] + [hidden_dim] * num_gcn_layers
        
        self.gcn_layers = nn.ModuleList(
            DenseGCNLayer(dims[i], dims[i+1]) for i in range(num_gcn_layers)
        ) # Spatial Layer
        
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True) # Temporal Layer
        self.head = nn.Linear(hidden_dim, horizon) # Final output layer
        self.dropout = dropout
    
    def forward(self, x:torch.Tensor, a_hat: torch.Tensor)-> torch.Tensor:
        h = x
        
        for layer in self.gcn_layers:
            h = F.relu(layer(h, a_hat))
            h = F.dropout(h, self.dropout, self.training)
        
        b, t, n, d = h.shape # b (batch size, timestep, nodes-stations) # temporal: GRU per station
        seq = h.permute(0,2,1,3).reshape(b*n, t, d)
        _, last_hidden = self.gru(seq)
        
        return self.head(last_hidden.squeeze(0)).view(b, n, -1)