import os
import numpy as np
import json
import torch

from source.utils.metrics import distinct
from source.utils.misc import sequence_mask
from source.inputter.batcher import create_turn_batch, create_kb_batch

class BeamGenerator(object):
    """
    BeamGenerator
    """
    def __init__(self,
                 model,
                 src_field,
                 tgt_field,
                 kb_field,
                 kbt_field,
                 beam_size=1,
                 max_length=10,
                 ignore_unk=True,
                 length_average=True,
                 use_gpu=False):
        self.model = model.cuda() if use_gpu else model
        self.src_field = src_field
        self.tgt_field = tgt_field
        self.kb_field = kb_field
        self.kbt_field = kbt_field
        self.k = beam_size
        self.max_length = max_length
        self.ignore_unk = ignore_unk
        self.length_average = length_average
        self.use_gpu = use_gpu
        self.PAD = tgt_field.stoi[tgt_field.pad_token]
        self.UNK = tgt_field.stoi[tgt_field.unk_token]
        self.BOS = tgt_field.stoi[tgt_field.bos_token]
        self.EOS = tgt_field.stoi[tgt_field.eos_token]
        self.V = self.tgt_field.vocab_size

    def forward(self, turn_inputs, kb_inputs, enc_hidden=None):
        """
        forward
        """
        # switch the model to evaluate mode
        self.model.eval()

        result_batch = []
        with torch.no_grad():
            self.model.reset_dlg_memory()
            self.model.load_kb_memory(kb_inputs)

            for i, input in enumerate(turn_inputs):
                if self.use_gpu:
                    input = input.cuda()
                
                self.model.reset_kb_memory()
                src, src_lengths = input.src
                tgt, tgt_lengths = input.tgt
                kb, kb_lengths = input.kb
                kb_gt, kb_gt_lengths = input.kb_gt

                dlg_enc_inputs = src[:, 1:-1], src_lengths - 2  # filter <bos> <eos>
                kb_enc_inputs = kb[:, 1:-1], kb_lengths - 2
                kb_gt_inputs = kb_gt[:, 1:-1], kb_gt_lengths - 2

                # generate word by word
                dlg_enc_hidden, dlg_enc_outputs, dlg_attn_mask = self.model.dlg_encode(dlg_enc_inputs, enc_hidden)
                enc_outputs, dec_init_state = self.model.kb_encode(kb_gt_inputs, dlg_enc_hidden, dlg_enc_outputs, dlg_attn_mask, kb_enc_inputs, enc_hidden)
                preds, lens, scores, dec_state = self.decode(dec_init_state)

                # update memory
                dec_inputs = tgt[:, :-1], tgt_lengths - 1  # filter <eos>
                self.model.reset_kb_memory()
                outputs = self.model.forward(dlg_enc_inputs=dlg_enc_inputs, kb_src_inputs=kb_enc_inputs, kb_tgt_inputs=kb_gt_inputs, dec_inputs=dec_inputs)
                self.model.update_memory(dlg_state_memory=outputs.dlg_state_memory, kbt_state_memory=outputs.kbt_state_memory)
                src_text = self.src_field.denumericalize(src)
                tgt_text = self.tgt_field.denumericalize(tgt)
                preds_text = self.tgt_field.denumericalize(preds)
                kb_gt_text = self.kb_field.denumericalize(kb_gt)
                conv_id = input.conv_id
                turn_label = input.turn_label

                scores = scores.tolist()

                ### debug
                #print(f'SRC: {src_text}')
                print(f'TGT: {tgt_text}')
                print(f'Pred: {preds_text}\n')

                enc_outputs.add(src=src_text, tgt=tgt_text, preds=preds_text, conv_id=conv_id, turn_label=turn_label, scores=scores, kb_gt=kb_gt_text)
                result_turn_batch = enc_outputs.flatten()
                result_batch += result_turn_batch

        return result_batch

    def decode(self, dec_state):
        """
        decode
        """
        long_tensor_type = torch.cuda.LongTensor if self.use_gpu else torch.LongTensor

        b = dec_state.get_batch_size()

        # [[0], [k*1], [k*2], ..., [k*(b-1)]]
        self.pos_index = (long_tensor_type(range(b)) * self.k).view(-1, 1)

        # Inflate the initial hidden states to be of size: (b*k, H)
        dec_state = dec_state.inflate(self.k)

        # Initialize the scores; for the first step,
        # ignore the inflated copies to avoid duplicate entries in the top k
        sequence_scores = long_tensor_type(b * self.k).float()
        sequence_scores.fill_(-float('inf'))
        sequence_scores.index_fill_(0, long_tensor_type(
            [i * self.k for i in range(b)]), 0.0)

        # Initialize the input vector
        input_var = long_tensor_type([self.BOS] * b * self.k)

        # Store decisions for backtracking
        stored_scores = list()
        stored_predecessors = list()
        stored_emitted_symbols = list()

        for t in range(1, self.max_length+1):
            # Run the RNN one step forward
            output, dec_state = self.model.decode(input_var, dec_state)
            log_softmax_output = output.squeeze(1)

            # To get the full sequence scores for the new candidates, add the
            # local scores for t_i to the predecessor scores for t_(i-1)
            sequence_scores = sequence_scores.unsqueeze(1).repeat(1, self.V)
            if self.length_average and t > 1:
                sequence_scores = sequence_scores * \
                    (1 - 1/t) + log_softmax_output / t
            else:
                sequence_scores += log_softmax_output

            scores, candidates = sequence_scores.view(b, -1).topk(self.k, dim=1)
            #print(candidates.dtype)
            # Reshape input = (b*k, 1) and sequence_scores = (b*k)
            input_var = (candidates % self.V)
            input_var = input_var.view(b * self.k)
            sequence_scores = scores.view(b * self.k)

            # Update fields for next timestep
            predecessors = (
                candidates / self.V + self.pos_index.expand_as(candidates)).view(b * self.k)

            #print(dec_state.dtype)
            predecessors = predecessors.long()
            #print(predecessors.dtype)
            dec_state = dec_state.index_select(predecessors)

            # Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
            stored_scores.append(sequence_scores.clone())
            eos_indices = input_var.data.eq(self.EOS)
            if eos_indices.nonzero().dim() > 0:
                sequence_scores.data.masked_fill_(eos_indices, -float('inf'))

            if self.ignore_unk:
                # Erase scores for UNK symbol so that they aren't expanded
                unk_indices = input_var.data.eq(self.UNK)
                if unk_indices.nonzero().dim() > 0:
                    sequence_scores.data.masked_fill_(unk_indices, -float('inf'))

            # Cache results for backtracking
            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(input_var)

        predicts, scores, lengths = self._backtrack(
            stored_predecessors, stored_emitted_symbols, stored_scores, b)

        predicts = predicts[:, :1]
        scores = scores[:, :1]
        lengths = long_tensor_type(lengths)[:, :1]
        mask = sequence_mask(lengths, max_len=self.max_length).eq(0)
        predicts[mask] = self.PAD

        return predicts, lengths, scores, dec_state

    def _backtrack(self, predecessors, symbols, scores, b):
        p = list()
        l = [[self.max_length] * self.k for _ in range(b)]

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = scores[-1].view(
            b, self.k).topk(self.k, dim=1)

        # initialize the sequence scores with the sorted last step beam scores
        s = sorted_score.clone()

        # the number of EOS found in the backward loop below for each batch
        batch_eos_found = [0] * b

        t = self.max_length - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add self.pos_index for indexing variable with b*k as the first dimension.
        t_predecessors = (
            sorted_idx + self.pos_index.expand_as(sorted_idx)).view(b * self.k)

        while t >= 0:
            # Re-order the variables with the back pointer
            current_symbol = symbols[t].index_select(0, t_predecessors)
            # Re-order the back pointer of the previous step with the back pointer of the current step
            t_predecessors = predecessors[t].index_select(0, t_predecessors)

            # This tricky block handles dropped sequences that see EOS earlier.
            # The basic idea is summarized below:
            #
            #   Terms:
            #       Ended sequences = sequences that see EOS early and dropped
            #       Survived sequences = sequences in the last step of the beams
            #
            #       Although the ended sequences are dropped during decoding,
            #   their generated symbols and complete backtracking information are still
            #   in the backtracking variables.
            #   For each batch, everytime we see an EOS in the backtracking process,
            #       1. If there is survived sequences in the return variables, replace
            #       the one with the lowest survived sequence score with the new ended
            #       sequences
            #       2. Otherwise, replace the ended sequence with the lowest sequence
            #       score with the new ended sequence
            #
            eos_indices = symbols[t].data.eq(self.EOS).nonzero()
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0)-1, -1, -1):
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = idx[0].item() // self.k
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = self.k - (batch_eos_found[b_idx] % self.k) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * self.k + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = predecessors[t][idx[0]]
                    current_symbol[res_idx] = symbols[t][idx[0]]
                    s[b_idx, res_k_idx] = scores[t][idx[0]]
                    l[b_idx][res_k_idx] = t + 1

            # record the back tracked results
            p.append(current_symbol)
            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        s, re_sorted_idx = s.topk(self.k)
        for b_idx in range(b):
            l[b_idx] = [l[b_idx][k_idx.item()]
                        for k_idx in re_sorted_idx[b_idx, :]]

        re_sorted_idx = (
            re_sorted_idx + self.pos_index.expand_as(re_sorted_idx)).view(b * self.k)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        predicts = torch.stack(p[::-1]).t()
        predicts = predicts[re_sorted_idx].contiguous().view(b, self.k, -1).data
        # p = [step.index_select(0, re_sorted_idx).view(b, self.k).data for step in reversed(p)]
        scores = s.data
        lengths = l

        return predicts, scores, lengths

    def generate(self, data_iter, output_dir=None, verbos=False):
        """
        generate
        """
        results = []
        num_batches = data_iter.n_batch
        for batch_idx in range(num_batches):
            local_data = data_iter.get_batch(batch_idx)
            turn_inputs = create_turn_batch(local_data['inputs'])
            kb_inputs = create_kb_batch(local_data['kbts'])
            assert len(turn_inputs) == local_data['max_turn']
            result_batch = self.forward(turn_inputs, kb_inputs, enc_hidden=None)
            results.append(result_batch)

        # post processing the results
        hyps = []
        refs = []
        conv_ids = []
        turn_labels = []
        srcs = []
        kb_gts = []

        for result_batch in results:
            for result_turn in result_batch:
                hyps.append(result_turn.preds[0].split(" "))
                refs.append(result_turn.tgt.split(" "))
                conv_ids.append(result_turn.conv_id)
                turn_labels.append(result_turn.turn_label)
                srcs.append(result_turn.src.split(" "))
                kb_gts.append(result_turn.kb_gt.split(" "))

        outputs = []

        for dia, tl, hyp, ref, src, kb_gt in zip(conv_ids, turn_labels, hyps, refs, srcs, kb_gts):
            hyp_str = " ".join(hyp)
            ref_str = " ".join(ref)
            src_str = " ".join(src)
            kb_gt_str = " ".join(kb_gt)
            sample = {"conv_id": str(dia),
                      "turn_label": str(tl),
                      "src": src_str,
                      "result": hyp_str,
                      "target": ref_str,
                      "tgt_kb": kb_gt_str}
            outputs.append(sample)

        avg_len = np.average([len(s) for s in hyps])
        intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(hyps)
        report_message = "Result:  Avg_Len={:.3f}   ".format(avg_len) + \
                         "Inter_Dist={:.4f}/{:.4f}".format(inter_dist1, inter_dist2)

        avg_len = np.average([len(s) for s in refs])
        intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(refs)
        target_message = "Target:  Avg_Len={:.3f}   ".format(avg_len) + \
                         "Inter_Dist={:.4f}/{:.4f}".format(inter_dist1, inter_dist2)

        message = target_message + "\n" + report_message
        if verbos:
            print(message)

        print("Dumping")
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open("%s/output.json" % output_dir, 'w', encoding='utf-8') as fw:
                json.dump(outputs, fw)
            
            f_res = open(f'{output_dir}/pred.txt', 'w', encoding='utf-8')
            f_tgt = open(f'{output_dir}/tgt.txt', 'w', encoding='utf-8')
            for sample in outputs:
                line1 = json.dumps(sample['result'])
                line2 = json.dumps(sample['target'])
                f_res.write(line1+"\n")
                f_tgt.write(line2+"\n")
            print("Saved generation results to '{}/output.json.'".format(output_dir))
