import torch
import torch.nn as nn
import os

from source.module.embedder import Embedder
from source.module.rnn_encoder import RNNEncoder, TransEncoder, transformers_model
from source.module.rnn_decoder import RNNDecoder
from source.utils.criterions import NLLLoss, MaskBCELoss
from source.utils.metrics import accuracy, perplexity
from source.utils.misc import Pack
from source.utils.misc import sequence_mask
from source.module.attention import Attention


class Seq2Seq(nn.Module):
    def __init__(self,src_field,tgt_field,kb_field,kbt_field,embed_size,hidden_size,padding_idx=None,num_attn_layers=2,
                num_rnn_layers=1,bidirectional=False,attn_mode="mlp",with_bridge=False,tie_embedding=False,
                 max_dlg_hop=1,max_kb_hop=1,dropout=0.0,use_gpu=False,pf_dim=2048, num_heads=8, bert_config=None, window_size=3):
        
        super(Seq2Seq, self).__init__()

        self.src_field = src_field
        self.tgt_field = tgt_field
        self.kb_field = kb_field
        self.kbt_field = kbt_field
        self.src_vocab_size = src_field.vocab_size
        self.tgt_vocab_size = tgt_field.vocab_size
        self.kb_vocab_size = kb_field.vocab_size
        self.kbt_vocab_size = kbt_field.vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.num_attn_layers = num_attn_layers
        self.num_rnn_layers = num_rnn_layers
        self.bidirectional = bidirectional
        self.attn_mode = attn_mode
        self.with_bridge = with_bridge
        self.tie_embedding = tie_embedding
        self.max_dlg_hop = max_dlg_hop
        self.max_kb_hop = max_kb_hop
        self.dropout = dropout
        self.use_gpu = use_gpu
        self.pf_dim = pf_dim
        self.num_heads = num_heads
        self.bert_config = bert_config
        self.window_size = window_size

        self.BOS = self.tgt_field.stoi[self.tgt_field.bos_token]

        self.enc_embedder = Embedder(num_embeddings=self.src_vocab_size,embedding_dim=self.embed_size,padding_idx=self.padding_idx)

        self.rnn_encoder = RNNEncoder(input_size=self.embed_size,hidden_size=self.hidden_size,
                                  embedder=self.enc_embedder,num_layers=self.num_rnn_layers,
                                  bidirectional=self.bidirectional,dropout=self.dropout)

        self.bert_encoder = transformers_model(self.bert_config, self.hidden_size, self.src_vocab_size)

        if self.with_bridge:
            self.bridge = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),nn.Tanh())

        if self.tie_embedding:
            assert self.src_vocab_size == self.tgt_vocab_size
            self.dec_embedder = self.enc_embedder
            self.kb_embedder = self.enc_embedder
            self.kbt_embedder = self.enc_embedder
        else:
            self.dec_embedder = Embedder(num_embeddings=self.tgt_vocab_size,embedding_dim=self.embed_size,padding_idx=self.padding_idx)
            self.kb_embedder = Embedder(num_embeddings=self.kb_vocab_size,embedding_dim=self.embed_size,padding_idx=self.padding_idx)
            self.kbt_embedder = Embedder(num_embeddings=self.kbt_vocab_size,embedding_dim=self.embed_size,padding_idx=self.padding_idx)

        # TODO: change KB to MLP transform
        self.trans_layer = nn.Linear(3 * self.embed_size, self.hidden_size, bias=True)

        # init memory
        self.dlg_state_memory = None
        self.dlg_history_memory = None
        self.dlg_history_index = None
        self.dlg_memory_masks = None
        self.kb_state_memory = None
        self.kb_history_memory = None
        self.kb_memory_masks = None
        self.kbt_state_memory = None
        self.kbt_history_index = None
        self.kbt_mask = None
        self.selector_mask = None
        self.dlg_lengths = []

        self.decoder = RNNDecoder(embedder=self.dec_embedder,max_dlg_hop=self.max_dlg_hop,max_kb_hop=self.max_kb_hop,
                                  input_size=self.embed_size,hidden_size=self.hidden_size,output_size=self.tgt_vocab_size,
                                  kbt_output_size=self.kbt_vocab_size,num_layers=self.num_rnn_layers,attn_mode=self.attn_mode,memory_size=self.hidden_size,
                                  kb_memory_size=self.hidden_size,dropout=self.dropout,padding_idx=self.padding_idx,
                                  use_gpu=self.use_gpu)
        self.sigmoid = nn.Sigmoid()

        self.src_attention = Attention(max_hop=self.max_dlg_hop,query_size=self.hidden_size,
                                   memory_size=self.hidden_size,hidden_size=self.hidden_size,
                                   num_layers=self.num_rnn_layers,dropout=self.dropout if self.num_rnn_layers > 1 else 0,
                                   mode=self.attn_mode,project=False)
        # loss definition
        if self.padding_idx is not None:
            weight = torch.ones(self.tgt_vocab_size)
            weight[self.padding_idx] = 0
        else:
            weight = None
        self.nll_loss = NLLLoss(weight=weight,ignore_index=self.padding_idx,reduction='mean')

        self.bce_loss = MaskBCELoss()

        if self.use_gpu:
            self.cuda()

    def load_kb_memory(self, kb_inputs):
        """
        load kb memory
        """
        kbs, kb_lengths = kb_inputs
        if self.use_gpu:
            kbs = kbs.cuda()
            kb_lengths = kb_lengths.cuda()

        batch_size, kb_num, kb_term = kbs.size()
        kbs = kbs[:, :, 1:-1]       # filter <bos> <eos>
        self.kbts = kbs

        # TODO: change kb_states
        kb_states = kbs
        kb_slots = kbs[:, :, -1]    # <object>
        kb_states = kb_states.contiguous().view(batch_size, kb_num, -1)  # (batch_size, kb_num, 3)
        kb_slots = kb_slots.contiguous().view(batch_size, kb_num)  # (batch_size, kb_num)
        self.kbt_slot_index = kb_slots

        kb_mask = kb_lengths.eq(0)
        self.kbt_mask = kb_mask      # (batch_size, kb_num)
        selector_mask = kb_mask.eq(0)
        self.selector_mask = selector_mask  # (batch_size, kb_num)
       
        embed_state = self.kbt_embedder(kb_states)
        embed_state = embed_state.contiguous().view(batch_size, kb_num, -1)
        #print('kbt embed', embed_state.shape)
        self.kbt_state_memory = self.trans_layer(embed_state)
        self.kbt_slot_memory = self.kbt_state_memory.clone()
    
    def __repr__(self):
        main_string = super(Seq2Seq, self).__repr__()
        num_parameters = sum([p.nelement() for p in self.parameters() if p.requires_grad])
        main_string += "\nNumber of parameters: {}\n".format(num_parameters)
        return main_string

    def save(self, filename):
        torch.save(self.state_dict(), filename)
        print("Saved model state to '{}'!".format(filename))

    def load(self, filename):
        if os.path.isfile(filename):
            state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
            self.load_state_dict(state_dict, strict=False)
            print("Loaded model state from '{}'".format(filename))
        else:
            print("Invalid model state file: '{}'".format(filename))
    
    def reset_dlg_memory(self):
        """
        reset memory
        """
        ## print('resetting memory')
        self.dlg_state_memory = None
        self.dlg_history_memory = None
        self.dlg_memory_masks = None
        self.dlg_history_index = None
        self.kbts = None
        self.dlg_lengths = []


    def reset_dlg_state_memory(self,turn_index):
        """
        reset memory
        """
        ## print('resetting memory')
        self.dlg_state_memory = self.dlg_state_memory[:,self.dlg_lengths[turn_index]:,:]
        self.dlg_history_memory = self.dlg_history_memory[:,self.dlg_lengths[turn_index]:,:]
        self.dlg_memory_masks = self.dlg_memory_masks[:,self.dlg_lengths[turn_index]:]
        self.dlg_history_index = self.dlg_history_index[:,self.dlg_lengths[turn_index]:]

    def reset_kb_memory(self):
        self.kb_state_memory = None
        self.kb_history_memory = None
        self.kb_history_index = None
        self.kb_memory_masks = None

    
    def make_src_mask(self, src):
        #src = [batch size, src len]
        src_mask = (src != self.padding_idx)
        src_mask = src_mask.long()
        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def dlg_encode(self, enc_inputs, hidden=None):
        outputs = Pack()

        inputs, lengths = enc_inputs
        # Trans encoder
        src_mask = self.make_src_mask(inputs)
        # print(inputs)
        # print(src_mask)
        enc_outputs = self.bert_encoder(inputs, src_mask)

        batch_size = enc_outputs.size(0)
        max_len = enc_outputs.size(1)
        attn_mask = sequence_mask(lengths, max_len).eq(0)
        enc_hidden = enc_outputs[:,-1:,:].squeeze(1).unsqueeze(0)

        if self.with_bridge:
            enc_hidden = self.bridge(enc_hidden)

        # insert dialog memory
        if self.dlg_state_memory is None:
            assert self.dlg_history_memory is None
            assert self.dlg_history_index is None
            assert self.dlg_memory_masks is None
            self.dlg_state_memory = enc_outputs
            self.dlg_history_memory = enc_outputs
            self.dlg_history_index = inputs
            self.dlg_memory_masks = attn_mask
        else:
                batch_state_memory = self.dlg_state_memory[:batch_size, :, :]
                self.dlg_state_memory = torch.cat([batch_state_memory, enc_outputs], dim=1)
                batch_history_memory = self.dlg_history_memory[:batch_size, :, :]
                self.dlg_history_memory = torch.cat([batch_history_memory, enc_outputs], dim=1)
                batch_history_index = self.dlg_history_index[:batch_size, :]
                self.dlg_history_index = torch.cat([batch_history_index, inputs], dim=-1)
                batch_memory_masks = self.dlg_memory_masks[:batch_size, :]
                self.dlg_memory_masks = torch.cat([batch_memory_masks, attn_mask], dim=-1)

        return enc_hidden, enc_outputs, attn_mask


    def kb_encode(self, kb_tgt_inputs, dlg_enc_hidden, dlg_enc_outputs, dlg_attn_mask, kb_src_inputs, hidden=None):
        kb_src_mask = self.make_src_mask(kb_src_inputs[0])
        kb_tgt_mask = self.make_src_mask(kb_tgt_inputs[0])

        kb_src_outputs = self.bert_encoder(kb_src_inputs[0], kb_src_mask)
        kb_tgt_outputs = self.bert_encoder(kb_tgt_inputs[0], kb_tgt_mask)
        kb_src_hidden = kb_src_outputs[:,-1:,:].squeeze(1).unsqueeze(0)
        kb_tgt_hidden = kb_tgt_outputs[:,-1:,:].squeeze(1).unsqueeze(0)
        
        outputs = Pack()
        inputs, lengths = kb_tgt_inputs
        batch_size = kb_tgt_outputs.size(0)
        max_len = kb_tgt_outputs.size(1)
        attn_mask = sequence_mask(lengths, max_len).eq(0)

        kb_src_inputs, kb_src_lengths = kb_src_inputs
        kb_src_batch_size = kb_src_inputs.size(0)
        kb_src_max_len = kb_src_inputs.size(1)
        kb_src_attn_mask = sequence_mask(kb_src_lengths, kb_src_max_len).eq(0)

        if self.with_bridge:
            enc_hidden = self.bridge(kb_tgt_hidden)

        dlg_weighted_hidden, _, _ = self.src_attention(query=kb_src_hidden[-1].unsqueeze(1),
                                                        key_memory=dlg_enc_outputs,
                                                        value_memory=dlg_enc_outputs,
                                                        hidden=kb_src_hidden,
                                                        mask=dlg_attn_mask)                                             

        # insert kb memory
        if self.kb_state_memory is None:
            assert self.kb_history_memory is None
            assert self.kb_history_index is None
            assert self.kb_memory_masks is None
            self.kb_state_memory = kb_tgt_outputs
            self.kb_history_memory = kb_tgt_outputs
            self.kb_history_index = inputs
            self.kb_memory_masks = attn_mask
        else:
            batch_state_memory = self.kb_state_memory[:batch_size, :, :]
            self.kb_state_memory = torch.cat([batch_state_memory, kb_tgt_outputs], dim=1)
            batch_history_memory = self.kb_history_memory[:batch_size, :, :]
            self.kb_history_memory = torch.cat([batch_history_memory, kb_tgt_outputs], dim=1)
            batch_history_index = self.kb_history_index[:batch_size, :]
            self.kb_history_index = torch.cat([batch_history_index, inputs], dim=-1)
            batch_memory_masks = self.kb_memory_masks[:batch_size, :]
            self.kb_memory_masks = torch.cat([batch_memory_masks, attn_mask], dim=-1)
            print("ck1")

        batch_kbt_inputs = self.kbts[:batch_size, :, :]
        batch_kbt_state_memory = self.kbt_state_memory[:batch_size, :, :]
        batch_kbt_slot_memory = self.kbt_slot_memory[:batch_size, :, :]
        batch_kbt_slot_index = self.kbt_slot_index[:batch_size, :]
        kbt_mask = self.kbt_mask[:batch_size, :]
        selector_mask = self.selector_mask[:batch_size, :]

        # create batched KB inputs
        kbt_memory, selector = self.decoder.initialize_kb(kb_inputs=batch_kbt_inputs, enc_hidden=dlg_enc_hidden)

        # initialize decoder state
        dec_init_state = self.decoder.initialize_state(
            hidden=dlg_weighted_hidden.squeeze(1).unsqueeze(0),
            dlg_state_memory=self.dlg_state_memory,
            dlg_history_memory=self.dlg_history_memory,
            dlg_attn_mask=self.dlg_memory_masks,
            dlg_history_index=self.dlg_history_index,
            kb_state_memory=self.kb_state_memory,
            kb_history_memory=self.kb_history_memory,
            kb_history_index=self.kb_history_index,
            kb_attn_mask=self.kb_memory_masks,
            kbt_memory=kbt_memory,
            kbt_state_memory=batch_kbt_state_memory,
            kbt_slot_memory=batch_kbt_slot_memory,
            kbt_slot_index=batch_kbt_slot_index,
            kbt_attn_mask=kbt_mask,
            selector=selector,
            selector_mask=selector_mask
        )

        return outputs, dec_init_state

    def decode(self, dec_inputs, state):
        prob, p_gen, p_con, pt_con, dlg_attn, kb_attn, kbt_attn, state = self.decoder.decode(dec_inputs, state)
        
        # logits copy from dialog history
        batch_size, max_len, word_size = prob.size()
        copy_index = state.dlg_history_index.unsqueeze(1).expand_as(dlg_attn).contiguous().view(batch_size, max_len, -1)
       
        copy_logits = dlg_attn.new_zeros(size=(batch_size, max_len, word_size),dtype=torch.float)
        copy_logits = copy_logits.scatter_add(dim=2, index=copy_index, src=dlg_attn)

        # logits copy from kb history
        kb_copy_index = state.kb_history_index.unsqueeze(1).expand_as(kb_attn).contiguous().view(batch_size, max_len, -1)
        kb_copy_logits = kb_attn.new_zeros(size=(batch_size, max_len, word_size),dtype=torch.float)
        kb_copy_logits = kb_copy_logits.scatter_add(dim=2, index=kb_copy_index, src=kb_attn)
        
        # logits copy from kbt
        index = state.kbt_slot_index[:batch_size, :].unsqueeze(1).expand_as(
            kbt_attn).contiguous().view(batch_size, max_len, -1)
        kbt_logits = kbt_attn.new_zeros(size=(batch_size, max_len, word_size),
                                      dtype=torch.float)
        kbt_logits = kbt_logits.scatter_add(dim=2, index=index, src=kbt_attn)

        # compute final distribution
        con_logits = p_gen * prob + (1 - p_gen) * copy_logits
        kb_logits = p_con * kb_copy_logits + (1 - p_con) * con_logits
        logits = pt_con * kbt_logits + (1 - pt_con) * kb_logits
        log_logits = torch.log(logits + 1e-12)

        return log_logits, state

    def forward(self, dlg_enc_inputs=None, kb_src_inputs=None, kb_tgt_inputs=None, dec_inputs=None, hidden=None):
        dlg_enc_hidden, dlg_enc_outputs, dlg_attn_mask = self.dlg_encode(dlg_enc_inputs, hidden)
        outputs, dec_init_state = self.kb_encode(kb_tgt_inputs, dlg_enc_hidden, dlg_enc_outputs, dlg_attn_mask, kb_src_inputs, hidden)
        
        #print('DEC IP', dec_inputs[0].size())
        prob, p_gen, p_con, pt_con, dlg_attn_probs, kb_attn_probs, kbt_prob, dec_state = self.decoder(dec_inputs, dec_init_state)

        # logits copy from dialog history
        batch_size, max_len, word_size = prob.size()
        copy_index = dec_init_state.dlg_history_index.unsqueeze(1).expand_as(dlg_attn_probs).contiguous().view(batch_size, max_len, -1)
        copy_logits = dlg_attn_probs.new_zeros(size=(batch_size, max_len, word_size),dtype=torch.float)
        copy_logits = copy_logits.scatter_add(dim=2, index=copy_index, src=dlg_attn_probs)

        
        # logits copy from kb history
        kb_copy_index = dec_init_state.kb_history_index.unsqueeze(1).expand_as(kb_attn_probs).contiguous().view(batch_size, max_len, -1)
        kb_copy_logits = kb_attn_probs.new_zeros(size=(batch_size, max_len, word_size),dtype=torch.float)
        kb_copy_logits = kb_copy_logits.scatter_add(dim=2, index=kb_copy_index, src=kb_attn_probs)
        
        index = dec_init_state.kbt_slot_index[:batch_size, :].unsqueeze(1).expand_as(
            kbt_prob).contiguous().view(batch_size, max_len, -1)

        kbt_logits = kbt_prob.new_zeros(size=(batch_size, max_len, word_size),
                                      dtype=torch.float)

        kbt_logits = kbt_logits.scatter_add(dim=2, index=index, src=kbt_prob)

        gate_logits = pt_con.squeeze(-1)
        selector_logits = dec_init_state.selector
        selector_mask = dec_init_state.selector_mask

        # compute final distribution
        con_logits = p_gen * prob + (1 - p_gen) * copy_logits
        kb_logits = p_con * kb_copy_logits + (1 - p_con) * con_logits
        logits = pt_con * kbt_logits + (1 - pt_con) * kb_logits
        log_logits = torch.log(logits + 1e-12)
        #log_logits = torch.log(kb_logits + 1e-12)

        outputs.add(logits=log_logits, gate_logits=gate_logits,
                    selector_logits=selector_logits, selector_mask=selector_mask,
                    dlg_state_memory=dec_state.dlg_state_memory,
                    kbt_state_memory=dec_state.kbt_state_memory)
        return outputs

    def collect_metrics(self, outputs, target):
        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)
        loss = 0

        # loss for generation
        logits = outputs.logits
        nll = self.nll_loss(logits, target)
        loss += nll

        acc = accuracy(logits, target, padding_idx=self.padding_idx)
        ppl, word_cnt = perplexity(logits, target, padding_idx=self.padding_idx)
        metrics.add(loss=loss, acc=acc)
        
        # For testing use this
        # metrics.add(loss=loss, acc=acc, pl=loss, vl=(loss,word_cnt))

        return metrics

    def update_memory(self, dlg_state_memory, kbt_state_memory):
        self.dlg_state_memory = dlg_state_memory
        self.kbt_state_memory = kbt_state_memory
       
    def iterate(self, turn_inputs, kbt_inputs, optimizer=None, grad_clip=None, entity_dir=None, is_training=True):
        self.reset_dlg_memory()

        self.load_kb_memory(kbt_inputs)
        metrics_list = []
        total_loss = 0

        for i, input in enumerate(turn_inputs):
            if self.use_gpu:
                input = input.cuda()
            self.reset_kb_memory()
            src, src_lengths = input.src
            tgt, tgt_lengths = input.tgt
            kb, kb_lengths = input.kb
            kb_gt, kb_gt_lengths = input.kb_gt

            
            dlg_enc_inputs = src[:, 1:-1], src_lengths - 2  # filter <bos> <eos>
            dec_inputs = tgt[:, :-1], tgt_lengths - 1  # filter <eos>
            target = tgt[:, 1:]  # filter <bos>
            kb_enc_inputs = kb[:, 1:-1], kb_lengths - 2   # filter <bos> <eos>
            kb_gt_enc_inputs = kb_gt[:, 1:-1], kb_gt_lengths - 2
            
            self.dlg_lengths.append(dlg_enc_inputs[0].size(1))
            if i>=self.window_size:
                self.reset_dlg_state_memory(i-self.window_size)

            outputs = self.forward(dlg_enc_inputs=dlg_enc_inputs, kb_src_inputs=kb_enc_inputs, kb_tgt_inputs=kb_gt_enc_inputs, dec_inputs=dec_inputs)
            metrics = self.collect_metrics(outputs, target)

            metrics_list.append(metrics)
            total_loss += metrics.loss
            self.update_memory(dlg_state_memory=outputs.dlg_state_memory, kbt_state_memory=outputs.kbt_state_memory)

        if torch.isnan(total_loss):
            raise ValueError("NAN loss encountered!")

        if is_training:
            assert optimizer is not None
            optimizer.zero_grad()
            total_loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=grad_clip)
            optimizer.step()

        return metrics_list
