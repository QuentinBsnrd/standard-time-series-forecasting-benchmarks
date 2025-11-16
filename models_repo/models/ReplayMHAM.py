import torch
import torch.nn as nn
import torch.nn.functional as F
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, input_size, projection_size, num_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=projection_size, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(projection_size) 
        self.ffn = nn.Sequential(  
            nn.Linear(projection_size, projection_size * 4),
            nn.ReLU(),
            nn.Linear(projection_size * 4, projection_size),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm(x + self.dropout(ffn_output))
        return x
    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        input_size = configs.enc_in
        projection_size = configs.d_model
        num_heads = configs.n_heads
        num_layers = 3

        self.input_projection = nn.Linear(input_size, projection_size)
        self.attention_layers = nn.ModuleList(
            [MultiHeadAttentionBlock(projection_size, projection_size, num_heads) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(projection_size, input_size)
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.input_projection(x)
        for layer in self.attention_layers:
            x = layer(x)
        attn_output = self.output_layer(x)
        pooled_attn = attn_output.max(dim=1)[0]
        scores = self.fc(pooled_attn).squeeze(-1)
        return attn_output, scores
