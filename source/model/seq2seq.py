# l# m

import torch
import torch.nn as nn
import os

from source.module.embedder import Embedder
from source.module.rnn_encoder import RNNEncoder
from source.module.rnn_decoder import RNNDecoder
from source.utils.criterions import NLLLoss, MaskBCELoss
from source.utils.metrics import accuracy
from source.utils.misc import Pack
from source.utils.misc import sequence_mask


class Seq2Seq(nn.Module):
    """
    Seq2Seq
    """
    def __init__(self,
                 src_field,
                 tgt_field,
                 kb_field,
                 embed_size,
                 hidden_size,
                 padding_idx=None,
                 num_layers=1,
                 bidirectional=False,
                 attn_mode="mlp",
                 with_bridge=False,
                 tie_embedding=False,
                 max_hop=1,
                 dropout=0.0,
                 use_gpu=False):
        super(Seq2Seq, self).__init__()

        self.src_field = src_field
        self.tgt_field = tgt_field
        self.kb_field = kb_field
        self.src_vocab_size = src_field.vocab_size
        self.tgt_vocab_size = tgt_field.vocab_size
        self.kb_vocab_size = kb_field.vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attn_mode = attn_mode
        self.with_bridge = with_bridge
        self.tie_embedding = tie_embedding
        self.max_hop = max_hop
        self.dropout = dropout
        self.use_gpu = use_gpu

        self.BOS = self.tgt_field.stoi[self.tgt_field.bos_token]
        #self.reward_fn_ = reward_fn

        self.enc_embedder = Embedder(num_embeddings=self.src_vocab_size,
                                     embedding_dim=self.embed_size,
                                     padding_idx=self.padding_idx)

        self.encoder = RNNEncoder(input_size=self.embed_size,
                                  hidden_size=self.hidden_size,
                                  embedder=self.enc_embedder,
                                  num_layers=self.num_layers,
                                  bidirectional=self.bidirectional,
                                  dropout=self.dropout)

        if self.with_bridge:
            self.bridge = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
            )

        if self.tie_embedding:
            assert self.src_vocab_size == self.tgt_vocab_size
            self.dec_embedder = self.enc_embedder
            self.kb_embedder = self.enc_embedder
        else:
            self.dec_embedder = Embedder(num_embeddings=self.tgt_vocab_size,
                                         embedding_dim=self.embed_size,
                                         padding_idx=self.padding_idx)
            self.kb_embedder = Embedder(num_embeddings=self.kb_vocab_size,
                                        embedding_dim=self.embed_size,
                                        padding_idx=self.padding_idx)

        # TODO: change KB to MLP transform
        self.trans_layer = nn.Linear(3 * self.embed_size, self.hidden_size, bias=True)

        # init memory
        self.dlg_state_memory = None
        self.dlg_history_memory = None
        self.dlg_history_index = None
        self.dlg_memory_masks = None
        self.kb_state_memory = None
        self.kb_history_memory = None
        self.kb_history_index = None
        self.kb_memory_masks = None

        self.decoder = RNNDecoder(embedder=self.dec_embedder,
                                  max_hop=self.max_hop,
                                  input_size=self.embed_size,
                                  hidden_size=self.hidden_size,
                                  output_size=self.tgt_vocab_size,
                                  num_layers=self.num_layers,
                                  attn_mode=self.attn_mode,
                                  memory_size=self.hidden_size,
                                  kb_memory_size=self.hidden_size,
                                  dropout=self.dropout,
                                  padding_idx=self.padding_idx,
                                  use_gpu=self.use_gpu)
        self.sigmoid = nn.Sigmoid()

        # loss definition
        if self.padding_idx is not None:
            weight = torch.ones(self.tgt_vocab_size)
            weight[self.padding_idx] = 0
        else:
            weight = None
        self.nll_loss = NLLLoss(weight=weight,
                                ignore_index=self.padding_idx,
                                reduction='mean')

        self.bce_loss = MaskBCELoss()

        if self.use_gpu:
            self.cuda()

    def __repr__(self):
        main_string = super(Seq2Seq, self).__repr__()
        num_parameters = sum([p.nelement() for p in self.parameters()])
        main_string += "\nNumber of parameters: {}\n".format(num_parameters)
        return main_string

    def save(self, filename):
        """
        save
        """
        torch.save(self.state_dict(), filename)
        print("Saved model state to '{}'!".format(filename))

    def load(self, filename):
        """
        load
        """
        if os.path.isfile(filename):
            state_dict = torch.load(
                filename, map_location=lambda storage, loc: storage)
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

    def reset_kb_memory(self):
        """
        reset memory
        """
        ## print('resetting memory')
        self.kb_state_memory = None
        self.kb_history_memory = None
        self.kb_history_index = None
        self.kb_memory_masks = None 

    '''def load_kb_memory(self, kb_inputs):
        """
        load kb memory
        """
        kbs, kb_lengths = kb_inputs
        if self.use_gpu:
            kbs = kbs.cuda()
            kb_lengths = kb_lengths.cuda()

        batch_size, kb_num, kb_term = kbs.size()
        kbs = kbs[:, :, 1:-1]       # filter <bos> <eos>
        self.kbs = kbs

        # TODO: change kb_states
        #kb_states = kbs[:, :, :-1]  # <subject, relation>
        kb_states = kbs
        kb_slots = kbs[:, :, -1]    # <object>
        kb_states = kb_states.contiguous().view(batch_size, kb_num, -1)  # (batch_size, kb_num, 3)
        kb_slots = kb_slots.contiguous().view(batch_size, kb_num)  # (batch_size, kb_num)
        self.kb_slot_index = kb_slots

        kb_mask = kb_lengths.eq(0)
        self.kb_mask = kb_mask      # (batch_size, kb_num)
        selector_mask = kb_mask.eq(0)
        self.selector_mask = selector_mask  # (batch_size, kb_num)

        embed_state = self.kb_embedder(kb_states)
        embed_state = embed_state.contiguous().view(batch_size, kb_num, -1)
        self.kb_state_memory = self.trans_layer(embed_state)
        self.kb_slot_memory = self.kb_state_memory.clone()
    '''
    def dlg_encode(self, enc_inputs, hidden=None):
        """
        encode
        """
        outputs = Pack()
        enc_outputs, enc_hidden = self.encoder(enc_inputs, hidden)
        # print('enc_',enc_hidden)
        # print(f"ENC IP: sh: {enc_inputs.size()}")
        #print(f"ENC IP: sh: {enc_inputs[0].size()}")
        #print(f"ENC OP: sh: {enc_outputs.size()}")
        inputs, lengths = enc_inputs
        batch_size = enc_outputs.size(0)
        max_len = enc_outputs.size(1)
        attn_mask = sequence_mask(lengths, max_len).eq(0)

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
            ## print("sizes before")
            #print(f"DSM: sh: {self.dlg_state_memory.shape}")
            #print(f"DHM: sh: {self.dlg_history_memory.shape}")
            batch_state_memory = self.dlg_state_memory[:batch_size, :, :]
            self.dlg_state_memory = torch.cat([batch_state_memory, enc_outputs], dim=1)
            batch_history_memory = self.dlg_history_memory[:batch_size, :, :]
            self.dlg_history_memory = torch.cat([batch_history_memory, enc_outputs], dim=1)
            batch_history_index = self.dlg_history_index[:batch_size, :]
            self.dlg_history_index = torch.cat([batch_history_index, inputs], dim=-1)
            batch_memory_masks = self.dlg_memory_masks[:batch_size, :]
            self.dlg_memory_masks = torch.cat([batch_memory_masks, attn_mask], dim=-1)

        ## print("sizes after")
        ## print(f"DSM: sh: {self.dlg_state_memory.shape}")
        ## print(f"DHM: sh: {self.dlg_history_memory.shape}")

        return enc_hidden


    def kb_encode(self, enc_inputs, dlg_enc_hidden, hidden=None):
        #print(f'kb_enc_inputs: {enc_inputs.shape}')
        kb_enc_outputs, kb_enc_hidden = self.encoder(enc_inputs, hidden)
        outputs = Pack()
        # print('enc_',kb_enc_hidden)
        # print(f"ENC IP: sh: {enc_inputs.size()}")
        #print(f"ENC IP: sh: {enc_inputs[0].size()}")
        #print(f"ENC OP: sh: {kb_enc_outputs.size()}")
        inputs, lengths = enc_inputs
        batch_size = kb_enc_outputs.size(0)
        max_len = kb_enc_outputs.size(1)
        attn_mask = sequence_mask(lengths, max_len).eq(0)

        if self.with_bridge:
            enc_hidden = self.bridge(kb_enc_hidden)

        # insert dialog memory
        if self.kb_state_memory is None:
            assert self.kb_history_memory is None
            assert self.kb_history_index is None
            assert self.kb_memory_masks is None
            self.kb_state_memory = kb_enc_outputs
            self.kb_history_memory = kb_enc_outputs
            self.kb_history_index = inputs
            self.kb_memory_masks = attn_mask
        else:
            ## print("sizes before")
            #print(f"DSM: sh: {self.dlg_state_memory.shape}")
            #print(f"DHM: sh: {self.dlg_history_memory.shape}")
            batch_state_memory = self.kb_state_memory[:batch_size, :, :]
            self.kb_state_memory = torch.cat([batch_state_memory, kb_enc_outputs], dim=1)
            batch_history_memory = self.kb_history_memory[:batch_size, :, :]
            self.kb_history_memory = torch.cat([batch_history_memory, kb_enc_outputs], dim=1)
            batch_history_index = self.kb_history_index[:batch_size, :]
            self.kb_history_index = torch.cat([batch_history_index, inputs], dim=-1)
            batch_memory_masks = self.kb_memory_masks[:batch_size, :]
            self.kb_memory_masks = torch.cat([batch_memory_masks, attn_mask], dim=-1)

        #print("sizes after")
        #print(f"KSM: sh: {self.kb_state_memory.shape}")
        #print(f"KHM: sh: {self.kb_history_memory.shape}")

        # initialize decoder state
        dec_init_state = self.decoder.initialize_state(
            hidden=dlg_enc_hidden,
            dlg_state_memory=self.dlg_state_memory,
            dlg_history_memory=self.dlg_history_memory,
            dlg_attn_mask=self.dlg_memory_masks,
            dlg_history_index=self.dlg_history_index,
            kb_state_memory=self.kb_state_memory,
            kb_history_memory=self.kb_history_memory,
            kb_history_index=self.kb_history_index,
            kb_attn_mask=self.kb_memory_masks
        )

        return outputs, dec_init_state

    def decode(self, dec_inputs, state):
        """
        decode
        """
        prob, state = self.decoder.decode(dec_inputs, state)
        log_logits = torch.log(prob + 1e-12)
        return log_logits, state

    def forward(self, enc_inputs, kb_enc_inputs, dec_inputs, hidden=None):
        """
        forward
        """
        dlg_enc_hidden = self.dlg_encode(enc_inputs, hidden)
        outputs, dec_init_state = self.kb_encode(kb_enc_inputs, dlg_enc_hidden, hidden)
        
        ## print('DEC IP', dec_inputs[0].size())
        prob, dec_state = self.decoder(dec_inputs, dec_init_state)
        ## print('output prob',prob.size())

        log_logits = torch.log(prob + 1e-12)

        outputs.add(logits=log_logits, dlg_state_memory=dec_state.dlg_state_memory)
        return outputs

    '''def sample(self, enc_inputs, dec_inputs, hidden=None, random_sample=False):
        """
        sampling for RL training
        """
        outputs, dec_init_state = self.encode(enc_inputs, hidden)

        batch_size, max_len = dec_inputs[0].size()
        pred_log_logits = torch.zeros((batch_size, max_len, self.tgt_vocab_size))  # zeros equal to padding idx
        pred_word = torch.ones((batch_size, max_len),
                               dtype=torch.long) * self.padding_idx  
        state = dec_init_state

        for i in range(max_len):
            if i == 0:
                dec_inputs = torch.ones((batch_size), dtype=torch.long) * self.BOS
                if self.use_gpu:
                    dec_inputs = dec_inputs.cuda()
                    pred_log_logits = pred_log_logits.cuda()
                    pred_word = pred_word.cuda()
            if i >= 1:
                if i == 1:
                    unfinish = dec_inputs.ne(self.padding_idx).long()
                else:
                    unfinish = unfinish * (dec_inputs.ne(self.padding_idx)).long()
                if unfinish.sum().item() == 0:
                    break

            log_logits, state = self.decode(dec_inputs, state)
            pred_log_logits[:, i:i + 1] = log_logits

            if random_sample:
                dec_inputs = torch.multinomial(torch.exp(log_logits.squeeze()), num_samples=1).view(-1)
            else:
                dec_inputs = torch.argmax(log_logits.squeeze(), dim=-1).view(-1)
            pred_word[:, i] = dec_inputs

        outputs.add(logits=pred_log_logits, pred_word=pred_word)
        return outputs

    def reward_fn(self, *args):
        return self.reward_fn_(self, *args)
'''
    def collect_metrics(self, outputs, target):
        """
        collect training metrics
        """
        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)
        loss = 0

        # loss for generation
        logits = outputs.logits
        nll = self.nll_loss(logits, target)
        loss += nll
        acc = accuracy(logits, target, padding_idx=self.padding_idx)
        metrics.add(loss=loss, acc=acc)

        return metrics

    def update_memory(self, dlg_state_memory):
        self.dlg_state_memory = dlg_state_memory
        ## print("Memory udpd\n")

    def iterate(self, turn_inputs,
                optimizer=None, grad_clip=None, use_rl=False, 
                entity_dir=None, is_training=True):
        """
        iterate
        """
        self.reset_dlg_memory()

        metrics_list = []
        total_loss = 0

        for i, input in enumerate(turn_inputs):
            if self.use_gpu:
                input = input.cuda()
            
            self.reset_kb_memory()
            #print(f'Dlg ip: {dlg_input.shape}\nKB: {kb_input.shape}')
            src, src_lengths = input.src
            tgt, tgt_lengths = input.tgt
            kb, kb_lengths = input.kb
            #print(f"[s2s]src: {src.shape}\ttgt: {tgt.shape}")

            
            dlg_enc_inputs = src[:, 1:-1], src_lengths - 2  # filter <bos> <eos>
            dec_inputs = tgt[:, :-1], tgt_lengths - 1  # filter <eos>
            target = tgt[:, 1:]  # filter <bos>
            kb_enc_inputs = kb[:, 1:-1], kb_lengths - 2   # filter <bos> <eos>

            #print(f'ENC: {dlg_enc_inputs}\n DEC: {dec_inputs}\nTGT: {target}\nKB: {kb_enc_inputs}')

            outputs = self.forward(dlg_enc_inputs, kb_enc_inputs, dec_inputs)
            metrics = self.collect_metrics(outputs, target)

            metrics_list.append(metrics)
            total_loss += metrics.loss
            # print(outputs.dialog_state_memory)
            # print(outputs.kb_state_memory)
            self.update_memory(dlg_state_memory=outputs.dlg_state_memory)

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
