import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Gated Residual Network to help for the flow of information
# Inspired from the Temporal Fusion Transformer Model
class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GatedLinearUnit, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.sigmoid(self.fc2(x))
        return x1 * x2

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(GatedResidualNetwork, self).__init__()
        self.glu1 = GatedLinearUnit(input_dim, hidden_dim)
        self.glu2 = GatedLinearUnit(hidden_dim, output_dim)
        self.skip_connection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        out = self.glu1(x)
        out = self.glu2(out)
        skip_out = self.skip_connection(x)
        return F.relu(out + skip_out)
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.register_buffer('pe', self._build_positional_encoding(max_len))

    def _build_positional_encoding(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

class SelfAttention(nn.Module):

    def __init__(self, d_model, nhead):
        super(SelfAttention, self).__init__()
        assert d_model % nhead == 0, "Embedding dimension must be 0 modulo number of heads."

        self.d_model = d_model
        self.num_heads = nhead
        self.head_dim = d_model // nhead
        self.qkv_proj = nn.Linear(d_model, 3*d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        
    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)
        
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.d_model)
        values = self.o_proj(values)

        return values ### we could visualize attention!!!

class CrossAttention(nn.Module):

    def __init__(self, d_model, nhead):
        super(CrossAttention, self).__init__()
        assert d_model % nhead == 0, "Embedding dimension must be 0 modulo number of heads."

        self.d_model = d_model
        self.num_heads = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2*d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        self.kv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, q, kv):
        batch_size, seq_length, _ = q.size()
        _, kv_seq_length, _ = kv.size()

        q = self.q_proj(q)
        kv = self.kv_proj(kv)

        # Separate K, V from linear output
        kv = kv.reshape(batch_size, kv_seq_length, self.num_heads, 2 * self.head_dim)
        kv = kv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        k, v = kv.chunk(2, dim=-1)

        # Process query tensor
        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]

        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.d_model)
        values = self.o_proj(values)

        return values

class Encoder(nn.Module):
    def __init__(self, d_model,time_window ,nhead, hidden_dim):
        super(Encoder, self).__init__()
        
        self.pos_encoding = PositionalEncoding(d_model, time_window)
        self.self_attn = SelfAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.gating = GatedResidualNetwork(d_model, d_model, hidden_dim)
        self.norm2 = nn.LayerNorm(d_model)
        self.feedforward = PositionwiseFeedForward(d_model, hidden_dim)

    def forward(self, x):
        x = self.pos_encoding(x)
        attn_output = self.self_attn(x)
        x = self.norm1(x + attn_output)
        gating_output = self.gating(x)
        x = self.norm2(x + gating_output)
        x = self.feedforward(x)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, time_window, nhead, hidden_dim):
        super(Decoder, self).__init__()

        self.pos_encoding = PositionalEncoding(d_model, time_window)
        self.self_attn = SelfAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.gating1 = GatedResidualNetwork(d_model, d_model, hidden_dim)
        self.norm2 = nn.LayerNorm(d_model)

        self.cross_attn = CrossAttention(d_model, nhead)
        self.norm3 = nn.LayerNorm(d_model)
        self.gating2 = GatedResidualNetwork(d_model, d_model, hidden_dim)
        self.norm4 = nn.LayerNorm(d_model)
        self.feedforward = PositionwiseFeedForward(d_model, hidden_dim)

    def forward(self, x, encoder_output):
        x = self.pos_encoding(x)
        self_attn_output = self.self_attn(x)
        x = self.norm1(x + self_attn_output)
        gating1_output = self.gating1(x)
        x = self.norm2(x + gating1_output)

        cross_attn_output = self.cross_attn(x, encoder_output)
        x = self.norm3(x + cross_attn_output)
        gating2_output = self.gating2(x)
        x = self.norm4(x + gating2_output)
        x = self.feedforward(x)
        return x

class Transformer1(nn.Module):
    def __init__(
        self, input_dim, output_dim, time_window, nhead=2, hidden_dim=256, layers=2):
        super(Transformer, self).__init__()
        self.lr = 0.0001
        self.conv_enc = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, stride=1, padding=2)
        self.layer_norm_enc = nn.BatchNorm1d(hidden_dim)
        self.conv_dec = nn.Conv1d(input_dim - 1, hidden_dim, kernel_size=5, stride=1, padding=2) # -1 to count for the actiosn minus one compare to the observaiton sequence
        self.layer_norm_dec = nn.BatchNorm1d(hidden_dim)

        self.encoder_layers = nn.ModuleList([Encoder(hidden_dim, time_window, nhead, hidden_dim) for _ in range(layers)])
        self.decoder_layers = nn.ModuleList([Decoder(hidden_dim, time_window, nhead, hidden_dim) for _ in range(layers)])
        
        self.linear1 = nn.Linear(hidden_dim * time_window, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, encoder_input, decoder_input):
        
        conv_enc = self.conv_enc(encoder_input.transpose(1, 2)).transpose(1, 2)
        conv_enc = self.layer_norm_enc(conv_enc.transpose(1, 2)).transpose(1, 2)
        memory = conv_enc
        for encoder_layer in self.encoder_layers:
            memory = encoder_layer(memory)
            
        conv_dec = self.conv_enc(decoder_input.transpose(1, 2)).transpose(1, 2)
        conv_dec = self.layer_norm_enc(conv_enc.transpose(1, 2)).transpose(1, 2)
        output = conv_dec
        for decoder_layer in self.decoder_layers:
            output = decoder_layer(output, memory)
        
        output = output.view(output.size(0), -1)
        output = self.linear1(output)
        output = self.activation(output)
        output = self.linear2(output).unsqueeze(1)        
        return output

class Transformer(nn.Module):
    def __init__(
        self, input_dim, output_dim, time_window, nhead=2, hidden_dim=256, layers=2):
        super(Transformer, self).__init__()
        self.lr = 0.0001
        self.linear_enc = nn.Linear(input_dim, hidden_dim)
        self.layer_norm_enc = nn.LayerNorm(hidden_dim)
        self.linear_dec = nn.Linear(input_dim, hidden_dim)
        self.layer_norm_dec = nn.LayerNorm(hidden_dim)

        self.encoder_layers = nn.ModuleList([Encoder(hidden_dim, time_window, nhead, hidden_dim) for _ in range(layers)])
        self.decoder_layers = nn.ModuleList([Decoder(hidden_dim, time_window, nhead, hidden_dim) for _ in range(layers)])
        
        #self.linear1 = nn.Linear(hidden_dim * time_window, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, encoder_input, decoder_input):
        linear_enc = self.linear_enc(encoder_input)
        linear_enc = self.layer_norm_enc(linear_enc)
        memory = linear_enc
        for encoder_layer in self.encoder_layers:
            memory = encoder_layer(memory)
            
        linear_dec = self.linear_dec(decoder_input)
        linear_dec = self.layer_norm_dec(linear_dec)
        output = linear_dec
        for decoder_layer in self.decoder_layers:
            output = decoder_layer(output, memory)
        #print(output.shape)
        #output = output.view(output.size(0), -1)
        #output = output.view(output.size(0), -1, output.size(2)).squeeze(1)

        output = self.linear1(output)
        output = self.activation(output)
        output = self.linear2(output)#.unsqueeze(1)        
        return output


