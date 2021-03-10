# m

import numpy as np
from torch.utils.data import Dataset

from source.utils.misc import Pack
from source.utils.misc import list2tensor


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
    """
    DialogBatcher
    """

    def __init__(self, batch_size, data_type="train", shuffle=False):
        self.batch_size = batch_size
        self.data_type = data_type
        self.shuffle = shuffle

        self.batch_data_list = []
        self.batch_size_list = []
        self.n_batch = None  # number of batches
        self.n_rows = None  # number of samples

    def __len__(self):
        return self.n_rows

    def prepare_epoch(self):
        if self.shuffle:
            np.random.shuffle(self.batch_data_list)

    def get_batch(self, batch_idx):
        local_data = self.batch_data_list[batch_idx]

        # print(f"local data [batcher]: {local_data}\nshape: {len(local_data)}\n")
        batch_data = self.create_batches(local_data)
        #print(f"batch data [batcher]: {batch_data}\nshape: {len(batch_data)}\n")

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

        # print(f"batch splits: {self.batch_data_list}")
        print('n_rows = %d, batch_size = %d, n_batch = %d.' % (self.n_rows, self.batch_size, self.n_batch))

    def create_batches(self, data):
        # sort by dialog turns
        sorted_data = sorted(data, key=lambda x: x['turns'], reverse=True)
        # print(f"sorted_data [batcher]: {sorted_data}\n")

        conv_ids = [sample['conv_id'] for sample in sorted_data]
        turns = [sample['turns'] for sample in sorted_data]
        #kbs = [sample['kb'] for sample in sorted_data]
        max_turn = max(turns)
        inputs = []
        for t in range(max_turn):
            turn_label = []
            turn_src = []
            turn_tgt = []
            turn_kb = []
            for sample in sorted_data:
                if sample['turns'] >= t+1:
                    turn_label.append(t+1)
                    turn_src.append(sample['src'][t])
                    turn_tgt.append(sample['tgt'][t])
                    turn_kb.append(sample['kb'][t])

            turn_batch_size = len(turn_src)
            conv_id = conv_ids[:turn_batch_size]

            assert len(turn_tgt) == turn_batch_size
            if self.data_type == "test":
                turn_input = {"conv_id": conv_id,
                              "turn_label": turn_label,
                              "src": turn_src,
                              "tgt": turn_tgt,
                              "kb": turn_kb
                              }
            else:
                turn_input = {"conv_id": conv_id, "turn_label": turn_label, "src": turn_src, "tgt": turn_tgt, "kb": turn_kb}
            inputs.append(turn_input)
            #print(f"{t} turn inputs [batcher]: src: {turn_input['src'][:20]}\ntgt: {turn_input['tgt'][:20]}\nKB: {turn_input['kb'][:20]}\n\n")
        batch_data = {
                      "max_turn": max_turn,
                      "inputs": inputs
                      }
        return batch_data


def create_turn_batch(data_list):
    """
    create_turn_batch
    """
    turn_batches = []
    for data_dict in data_list:
        batch = Pack()
        for key in data_dict.keys():
            if key in ['src', 'tgt', 'kb']:
                batch[key] = list2tensor([x for x in data_dict[key]])
            else:
                batch[key] = data_dict[key]
        turn_batches.append(batch)
    # print(f"turns [batcher]: {turn_batches}")
    return turn_batches


def create_kb_batch(kb_list):
    """
    create_kb_batch
    """
    kb_batches = list2tensor(kb_list)
    return kb_batches
