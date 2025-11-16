import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Simple Multihead Temporal Attention
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        input_size = configs.input_size
        projection_size = configs.d_model
        self.num_heads = configs.n_heads
        output_size = configs.input_size
        dropout = configs.dropout
        self.head_size = projection_size // self.num_heads

        # Linear layers for queries, keys, and values for each head
        self.query_layers = nn.ModuleList([nn.Linear(input_size, self.head_size) for _ in range(self.num_heads)])
        self.key_layers = nn.ModuleList([nn.Linear(input_size, self.head_size) for _ in range(self.num_heads)])
        self.value_layers = nn.ModuleList([nn.Linear(input_size, self.head_size) for _ in range(self.num_heads)])
        
        # Final linear layer to combine outputs from different heads
        self.fc = nn.Linear(projection_size, projection_size*self.num_heads)
        # Temporal up-sampling (pour ajuster la séquence)
        self.temporal_projector = nn.Linear(configs.seq_len, configs.pred_len)
        
        self.output_layer = nn.Linear(projection_size, output_size)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        combined_heads = []
        
        for i in range(self.num_heads): # Method described in the paper: Attention Is All You Need.
            query = self.query_layers[i](inputs)
            key = self.key_layers[i](inputs)
            value = self.value_layers[i](inputs)
            
            # Compute attention scores
            attention_scores = torch.bmm(query, key.transpose(1, 2)) / (self.head_size)
            attention_weights = F.softmax(attention_scores, dim=-1)
            # print("attention_weights",attention_weights.shape)
            
            # Apply attention weights to values
            context = torch.bmm(attention_weights, value) # shape: (batch_size, head_size, seq_len)
            context = self.dropout(context)
            combined_heads.append(context)
            
        # Concatenate outputs from different heads
        combined_heads = torch.cat(combined_heads, dim=-1)

        # Project the temporal dimension to match the desired output sequence length
        combined_heads = combined_heads.transpose(1, 2)  # (batch_size, projection_size, seq_len)
        upsampled = self.temporal_projector(combined_heads)  # (batch_size, projection_size, pred_len)
        upsampled = upsampled.transpose(1, 2)  # (batch_size, pred_len, projection_size)
        
        # Final output projection
        output = self.output_layer(upsampled)  # (batch_size, pred_len, input_size)
        return output