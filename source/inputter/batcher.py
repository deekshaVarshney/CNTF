import numpy as np
from torch.utils.data import Dataset

from source.utils.misc import list2tensor
from source.utils.misc import Pack

bert_max_len = 512

class DialogDataset(Dataset):
    """
    DialogDataset
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DialogBatcher(object):

    def __init__(self, batch_size, data_type="train", shuffle=False):
        self.batch_size = batch_size
        self.data_type = data_type
        self.shuffle = shuffle

        self.batch_data_list = []
        self.batch_size_list = []
        self.n_batch = None 
        self.n_rows = None 

    def __len__(self):
        return self.n_rows

    def prepare_epoch(self):
        if self.shuffle:
            np.random.shuffle(self.batch_data_list)

    def get_batch(self, batch_idx):
        local_data = self.batch_data_list[batch_idx]
        batch_data = self.create_batches(local_data)
        return batch_data

    def prepare_input_list(self, input_data_list):
        if self.shuffle:
            np.random.shuffle(input_data_list)

        self.n_rows = remain_rows = len(input_data_list)
        while remain_rows > 0:
            self.batch_data_list.append({})
            active_size = min(remain_rows, self.batch_size)
            self.batch_size_list.append(active_size)
            remain_rows -= active_size
        self.n_batch = len(self.batch_size_list)

        for batch_idx in range(self.n_batch):
            st_idx = batch_idx * self.batch_size
            ed_idx = st_idx + self.batch_size
            # print(f"st {st_idx}, en {ed_idx}")
            local_batch_input = input_data_list[st_idx: ed_idx]
            self.batch_data_list[batch_idx] = local_batch_input

        print('n_rows = %d, batch_size = %d, n_batch = %d.' % (self.n_rows, self.batch_size, self.n_batch))

    def create_batches(self, data):
        # sort by dialog turns
        sorted_data = sorted(data, key=lambda x: x['turns'], reverse=True)

        conv_ids = [sample['conv_id'] for sample in sorted_data]
        turns = [sample['turns'] for sample in sorted_data]
        kbts = [sample['kbt'] for sample in sorted_data]
        max_turn = max(turns)
        inputs = []
        for t in range(max_turn):
            turn_label = []
            turn_src = []
            turn_tgt = []
            turn_kb = []
            turn_kb_gt = []
            turn_entity = []
            turn_ptr = []
            turn_kb_ptr = []
            for sample in sorted_data:
                if sample['turns'] >= t+1:
                    turn_label.append(t+1)
                    turn_src.append(sample['src'][t][:bert_max_len])
                    turn_tgt.append(sample['tgt'][t][:bert_max_len])
                    turn_kb.append(sample['kb'][t][:bert_max_len])
                    turn_kb_gt.append(sample['kb_gt'][t][:bert_max_len])

            turn_batch_size = len(turn_src)
            conv_id = conv_ids[:turn_batch_size]

            assert len(turn_tgt) == turn_batch_size
            if self.data_type == "test":
                turn_input = {"conv_id": conv_id,"turn_label": turn_label,"src": turn_src,
                               "tgt": turn_tgt,"kb": turn_kb,"kb_gt": turn_kb_gt}
            else:
                turn_input = {"conv_id": conv_id, "turn_label": turn_label, "src": turn_src, 
                              "tgt": turn_tgt, "kb": turn_kb, "kb_gt": turn_kb_gt}
            inputs.append(turn_input)
            
        batch_data = {"max_turn": max_turn,"inputs": inputs, "kbts": kbts}
        '''for i in kbts:
            print('batcher kbt', i[:10])
            print('\n')'''
        return batch_data


def create_turn_batch(data_list):
    """
    create_turn_batch
    """
    turn_batches = []
    for data_dict in data_list:
        batch = Pack()
        for key in data_dict.keys():
            if key in ['src', 'tgt', 'kb', 'kb_gt', 'ptr_index', 'kb_index']:
                batch[key] = list2tensor([x for x in data_dict[key]])
            else:
                batch[key] = data_dict[key]
        turn_batches.append(batch)
    return turn_batches

def create_kb_batch(kb_list):
    """
    create_kb_batch
    """
    new_kb_list = []
    for i in kb_list:
        kbt = []
        for j in i:
            if len(j) == 6:
                del j[2]
            elif len(j) == 7:
                #print(j)
                del j[2:4]
            if len(j) == 5:
                kbt.append(j)
        new_kb_list.append(kbt)
    kb_batches = list2tensor(new_kb_list)
    #print(kb_batches)
    return kb_batches