import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from transformers import BertModel, BertConfig


class RNNEncoder(nn.Module):
    """
    A GRU recurrent neural network encoder.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 embedder=None,
                 num_layers=1,
                 bidirectional=True,
                 dropout=0.0):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        rnn_hidden_size = hidden_size // num_directions

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.embedder = embedder
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.rnn_hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          bidirectional=self.bidirectional)
        
        map(nn.init.orthogonal_, self.rnn.all_weights)
    
    def forward(self, inputs, hidden=None):
        """
        forward
        """
        if isinstance(inputs, tuple):
            inputs, lengths = inputs
        else:
            inputs, lengths = inputs, None

        if self.embedder is not None:
            rnn_inputs = self.embedder(inputs)
        else:
            rnn_inputs = inputs

        batch_size = rnn_inputs.size(0)

        if lengths is not None:
            num_valid = lengths.gt(0).int().sum().item()
            sorted_lengths, indices = lengths.sort(descending=True)
            rnn_inputs = rnn_inputs.index_select(0, indices)

            rnn_inputs = pack_padded_sequence(
                rnn_inputs[:num_valid],
                sorted_lengths[:num_valid].tolist(),
                batch_first=True)

            if hidden is not None:
                hidden = hidden.index_select(1, indices)[:, :num_valid]

        outputs, last_hidden = self.rnn(rnn_inputs, hidden)

        if self.bidirectional:
            last_hidden = self._bridge_bidirectional_hidden(last_hidden)

        if lengths is not None:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)

            if num_valid < batch_size:
                zeros = outputs.new_zeros(
                    batch_size - num_valid, outputs.size(1), self.hidden_size)
                outputs = torch.cat([outputs, zeros], dim=0)

                zeros = last_hidden.new_zeros(
                    self.num_layers, batch_size - num_valid, self.hidden_size)
                last_hidden = torch.cat([last_hidden, zeros], dim=1)

            _, inv_indices = indices.sort()
            outputs = outputs.index_select(0, inv_indices)
            last_hidden = last_hidden.index_select(1, inv_indices)

        return outputs, last_hidden

    @staticmethod
    def _bridge_bidirectional_hidden(hidden):
        """
        the bidirectional hidden is (num_layers * num_directions, batch_size, hidden_size)
        we need to convert it to (num_layers, batch_size, num_directions * hidden_size)
        """
        num_layers = hidden.size(0) // 2
        _, batch_size, hidden_size = hidden.size()
        return hidden.view(num_layers, 2, batch_size, hidden_size)\
            .transpose(1, 2).contiguous().view(num_layers, batch_size, hidden_size * 2)


class HRNNEncoder(nn.Module):
    """
    HRNNEncoder
    """
    def __init__(self,
                 sub_encoder,
                 hiera_encoder):
        super(HRNNEncoder, self).__init__()
        self.sub_encoder = sub_encoder
        self.hiera_encoder = hiera_encoder

    def forward(self, inputs, features=None, sub_hidden=None, hiera_hidden=None,
                return_last_sub_outputs=False):
        """
        inputs: Tuple[Tensor(batch_size, max_hiera_len, max_sub_len), 
                Tensor(batch_size, max_hiera_len)]
        """
        indices, lengths = inputs
        batch_size, max_hiera_len, max_sub_len = indices.size()
        hiera_lengths = lengths.gt(0).long().sum(dim=1)

        # Forward of sub encoder
        indices = indices.view(-1, max_sub_len)
        sub_lengths = lengths.view(-1)
        sub_enc_inputs = (indices, sub_lengths)
        sub_outputs, sub_hidden = self.sub_encoder(sub_enc_inputs, sub_hidden)
        sub_hidden = sub_hidden[-1].view(batch_size, max_hiera_len, -1)

        if features is not None:
            sub_hidden = torch.cat([sub_hidden, features], dim=-1)

        # Forward of hiera encoder
        hiera_enc_inputs = (sub_hidden, hiera_lengths)
        hiera_outputs, hiera_hidden = self.hiera_encoder(
            hiera_enc_inputs, hiera_hidden)

        if return_last_sub_outputs:
            sub_outputs = sub_outputs.view(
                batch_size, max_hiera_len, max_sub_len, -1)
            last_sub_outputs = torch.stack(
                [sub_outputs[b, l - 1] for b, l in enumerate(hiera_lengths)])
            last_sub_lengths = torch.stack(
                [lengths[b, l - 1] for b, l in enumerate(hiera_lengths)])
            max_len = last_sub_lengths.max()
            last_sub_outputs = last_sub_outputs[:, :max_len]
            return hiera_outputs, hiera_hidden, (last_sub_outputs, last_sub_lengths)
        else:
            return hiera_outputs, hiera_hidden, None

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).cuda()
        
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
    
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        #print(f'query {Q.shape}')
        K = self.fc_k(key)
        #print(f'kye {K.shape}')
        V = self.fc_v(value)
        #print(f'value {V.shape}')
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x


class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    # def __getattr__(self, name):
    #     try:
    #         return super().__getattr__(name)
    #     except AttributeError:
    #         return getattr(self.module, name)
    
    def forward(self, src, src_mask):
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        #print(f"TRANS LAYER end: src {src.shape}")
        return src

class TransEncoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim=256, 
                 n_layers=6, 
                 n_heads=8, 
                 pf_dim=2048,
                 dropout=0, 
                 embedder=None):
        super().__init__()
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.embedder = embedder
        self.pos_embedding = nn.Embedding(input_dim, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout)
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).cuda()
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        
        src, lens = src
        batch_size = src.size(0)
        src_len = src.size(1)
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).cuda()
        
        #pos = [batch size, src len]
        
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        return src


class transformers_model(nn.Module):
    def __init__(self, config, hidden_size, vocab_size):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        encoder_config = BertConfig.from_json_file(self.config)
        self.encoder = BertModel(encoder_config)

    def forward(self, input_ids, mask_encoder_input):
        encoder_hidden_states = self.encoder(input_ids, mask_encoder_input)
        encoder_hidden_states = encoder_hidden_states.last_hidden_state
        return encoder_hidden_states