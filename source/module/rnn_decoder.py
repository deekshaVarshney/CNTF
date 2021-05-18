import torch
import torch.nn as nn

from source.module.attention import Attention
from source.module.decoder_state import DecoderState
from source.utils.misc import sequence_mask
from source.module.memory_helper import KnowledgeMemoryv3


class RNNDecoder(nn.Module):
    """
    A GRU recurrent neural network decoder.
    """
    def __init__(self,
                 embedder,
                 max_dlg_hop,
                 max_kb_hop,
                 input_size,
                 hidden_size,
                 output_size,
                 kbt_output_size,
                 num_layers=1,
                 attn_mode="mlp",
                 memory_size=None,
                 kb_memory_size=None,
                 dropout=0.0,
                 padding_idx=None,
                 use_gpu=False):
        super(RNNDecoder, self).__init__()

        self.embedder = embedder
        self.max_dlg_hop = max_dlg_hop
        self.max_kb_hop = max_kb_hop
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.kbt_output_size = kbt_output_size
        self.num_layers = num_layers
        self.attn_mode = attn_mode
        self.memory_size = memory_size
        self.kb_memory_size = kb_memory_size
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.use_gpu = use_gpu

        self.rnn_input_size = self.input_size
        self.out_input_size = self.input_size + self.hidden_size
        self.kb_input_size =  self.hidden_size + self.kb_memory_size
        self.kbt_input_size = self.kb_memory_size

        self.rnn_input_size += self.memory_size + 2*self.kb_memory_size
        self.out_input_size += self.memory_size + 2*self.kb_memory_size
        self.dlg_attention = Attention(max_hop=self.max_dlg_hop,
                                   query_size=self.hidden_size,
                                   memory_size=self.memory_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.num_layers,
                                   dropout=self.dropout if self.num_layers > 1 else 0,
                                   mode=self.attn_mode,
                                   project=False)
        self.kb_attention = Attention(max_hop=self.max_kb_hop,
                            query_size=self.hidden_size,
                            memory_size=self.memory_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            dropout=self.dropout if self.num_layers > 1 else 0,
                            mode=self.attn_mode,
                            project=False)
        
        self.kb_memory_v3 = KnowledgeMemoryv3(vocab_size=self.kbt_output_size,
                                              query_size=self.hidden_size,
                                              memory_size=self.memory_size,
                                              max_hop=self.max_kb_hop,
                                              padding_idx=self.padding_idx,
                                              use_gpu=self.use_gpu)

        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          batch_first=True)
        map(nn.init.orthogonal_, self.rnn.all_weights)

        self.gate_layer = nn.Sequential(
            nn.Linear(self.out_input_size, 1, bias=True),
            nn.Sigmoid()
        )
        self.copy_gate_layer = nn.Sequential(
            nn.Linear(self.kb_input_size, 1, bias=True),
            nn.Sigmoid()
        )
        self.copy_gate_layer_kbt = nn.Sequential(
            nn.Linear(self.kbt_input_size, 1, bias=True),
            nn.Sigmoid()
        )

        if self.out_input_size > self.hidden_size:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.output_size),
                nn.Softmax(dim=-1),
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.output_size),
                nn.Softmax(dim=-1),
            )

    def initialize_kb(self, kb_inputs, enc_hidden):
        kb_memory, selector = self.kb_memory_v3.load_memory(kb_inputs, enc_hidden)
        return kb_memory, selector

    @staticmethod
    def initialize_state(hidden,
                         dlg_state_memory=None,
                         dlg_history_memory=None,
                         dlg_history_index=None,
                         dlg_attn_mask=None,
                         kb_state_memory=None,
                         kb_history_memory=None,
                         kb_history_index=None,
                         kb_attn_mask=None,
                         kbt_memory=None,
                         kbt_state_memory=None,
                         kbt_slot_memory=None,
                         kbt_slot_index=None,
                         kbt_attn_mask=None,
                         selector=None,
                         selector_mask=None):
        """
        initialize_state
        """
        init_state = DecoderState(
            hidden=hidden,
            dlg_state_memory=dlg_state_memory,
            dlg_history_memory=dlg_history_memory,
            dlg_history_index=dlg_history_index,
            dlg_attn_mask=dlg_attn_mask,
            kb_state_memory=kb_state_memory,
            kb_history_memory=kb_history_memory,
            kb_history_index=kb_history_index,
            kb_attn_mask=kb_attn_mask,
            kbt_memory=kbt_memory,
            kbt_state_memory=kbt_state_memory,
            kbt_slot_memory=kbt_slot_memory,
            kbt_slot_index=kbt_slot_index,
            kbt_attn_mask=kbt_attn_mask,
            selector=selector,
            selector_mask=selector_mask
        )

        return init_state

    def decode(self, inputs, state, is_training=False):
        """
        decode
        """
        rnn_input_list = []
        out_input_list = []
        kb_input_list = []
        kbt_input_list = []
        
        inputs = self.embedder(inputs)

        ## shape: (batch_size, 1, input_size)
        inputs = inputs.unsqueeze(1)
        rnn_input_list.append(inputs)
        out_input_list.append(inputs)

        hidden = state.hidden
        # getting hidden of last layer
        query = hidden[-1].unsqueeze(1)
        #print(f"query sh: {query.shape}")
        # [bs 1 hs]


        dlg_weighted_context, dlg_attn, dlg_updated_memory = self.dlg_attention(query=query,
                                                                key_memory=state.dlg_state_memory.clone(),
                                                                value_memory=state.dlg_history_memory.clone(),
                                                                hidden=hidden,
                                                                mask=state.dlg_attn_mask)
        
        # attention bw all utterance represntations and current KB
        kb_weighted_context, kb_attn, kb_updated_memory = self.kb_attention(query=query,
                                                                key_memory=state.kb_state_memory.clone(),
                                                                value_memory=state.kb_history_memory.clone(),
                                                                hidden=hidden,
                                                                mask=state.kb_attn_mask)
        
        # generate from kbt
        weighted_kbt, kbt_attn = self.kb_memory_v3(query=query,
                                                 kb_memory_db=state.kbt_memory,
                                                 selector=None,
                                                 mask=state.kbt_attn_mask)
        
        rnn_input_list.append(dlg_weighted_context)
        out_input_list.append(dlg_weighted_context)

        rnn_input_list.append(kb_weighted_context)
        out_input_list.append(kb_weighted_context)

        # kbt append
        rnn_input_list.append(weighted_kbt)
        out_input_list.append(weighted_kbt)

        kb_input_list.append(kb_weighted_context)
        kbt_input_list.append(weighted_kbt)

        state.dlg_state_memory = dlg_updated_memory.clone()
        state.kb_state_memory = kb_updated_memory.clone()

        rnn_input = torch.cat(rnn_input_list, dim=-1)
        rnn_output, new_hidden = self.rnn(rnn_input, hidden)
        
        out_input_list.append(rnn_output)
        kb_input_list.append(rnn_output)
        out_input = torch.cat(out_input_list, dim=-1)
        kb_input = torch.cat(kb_input_list, dim=-1)
        kbt_input = torch.cat(kbt_input_list, dim=-1)
        state.hidden = new_hidden

        if is_training:
            return out_input, kb_input, kbt_input, dlg_attn, kb_attn, kbt_attn, state
        else:
            prob = self.output_layer(out_input)
            p_gen = self.gate_layer(out_input)
            p_con = self.copy_gate_layer(kb_input)
            pt_con = self.copy_gate_layer_kbt(kbt_input)
            return prob, p_gen, p_con, pt_con, dlg_attn, kb_attn, kbt_attn, state

    def forward(self, dec_inputs, state):
        """
        forward
        """
        inputs, lengths = dec_inputs
        ## lengths Contains sizes of tgt sentences in that turn of a batch

        batch_size, max_len = inputs.size()

        out_inputs = inputs.new_zeros(
            size=(batch_size, max_len, self.out_input_size),
            dtype=torch.float)

        kb_inputs = inputs.new_zeros(
            size=(batch_size, max_len, self.kb_input_size),
            dtype=torch.float)

        kbt_inputs = inputs.new_zeros(
            size=(batch_size, max_len, self.kbt_input_size),
            dtype=torch.float)

        #print('init kb', self.kbt_input_size)

        out_dlg_attn_size = state.dlg_history_memory.size(1)
        out_dlg_attn_probs = inputs.new_zeros(
            size=(batch_size, max_len, out_dlg_attn_size),
            dtype=torch.float)

        out_kb_attn_size = state.kb_history_memory.size(1)
        out_kb_attn_probs = inputs.new_zeros(
            size=(batch_size, max_len, out_kb_attn_size),
            dtype=torch.float)
        
        out_kbt_attn_size = state.kbt_slot_memory.size(1)
        out_kbt_attn_probs = inputs.new_zeros(
            size=(batch_size, max_len, out_kbt_attn_size),
            dtype=torch.float)

        ## sort by lengths
        sorted_lengths, indices = lengths.sort(descending=True)
        inputs = inputs.index_select(0, indices)
        state = state.index_select(indices)

        ## number of valid inputs (i.e. not padding index) in each time step
        num_valid_list = sequence_mask(sorted_lengths).int().sum(dim=0)

        for i, num_valid in enumerate(num_valid_list):
            dec_input = inputs[:num_valid, i]
            valid_state = state.slice_select(num_valid)

            ## decode for one step
            out_input, kb_input, kbt_input, dlg_attn, kb_attn, kbt_attn, valid_state = self.decode(dec_input, valid_state, is_training=True)
            state.hidden[:, :num_valid] = valid_state.hidden
            state.dlg_state_memory[:num_valid, :, :] = valid_state.dlg_state_memory
            state.kb_state_memory[:num_valid, :, :] = valid_state.kb_state_memory
            state.kbt_state_memory[:num_valid, :, :] = valid_state.kbt_state_memory

            out_inputs[:num_valid, i] = out_input.squeeze(1)
            kb_inputs[:num_valid, i] = kb_input.squeeze(1)
            kbt_inputs[:num_valid, i] = kbt_input.squeeze(1)
            out_dlg_attn_probs[:num_valid, i] = dlg_attn.squeeze(1)
            out_kb_attn_probs[:num_valid, i] = kb_attn.squeeze(1)
            out_kbt_attn_probs[:num_valid, i] = kbt_attn.squeeze(1)


        ## resort
        _, inv_indices = indices.sort()
        state = state.index_select(inv_indices)
        out_inputs = out_inputs.index_select(0, inv_indices)
        kb_inputs = kb_inputs.index_select(0, inv_indices)
        kbt_inputs = kbt_inputs.index_select(0, inv_indices)
        dlg_attn_probs = out_dlg_attn_probs.index_select(0, inv_indices)
        kb_attn_probs = out_kb_attn_probs.index_select(0, inv_indices)
        kbt_attn_probs = out_kbt_attn_probs.index_select(0, inv_indices)

        probs = self.output_layer(out_inputs)
        p_gen = self.gate_layer(out_inputs)
        p_con = self.copy_gate_layer(kb_inputs)
        pt_con = self.copy_gate_layer_kbt(kbt_inputs)

        return probs, p_gen, p_con, pt_con, dlg_attn_probs, kb_attn_probs, kbt_attn_probs, state
