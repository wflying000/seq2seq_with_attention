import torch
from torch.utils.data import Dataset

class NMTDataset(Dataset):
    def __init__(
        self, 
        src_data, 
        tgt_data, 
        src_word2id, 
        tgt_word2id,
        src_pad_id,
        tgt_pad_id,
        src_unk_id,
        tgt_unk_id,
    ):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.src_word2id = src_word2id
        self.tgt_word2id = tgt_word2id
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.src_unk_id = src_unk_id
        self.tgt_unk_id = tgt_unk_id

    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        src_sent = self.src_data[idx] 
        tgt_sent = self.tgt_data[idx]

        src_ids = [self.src_word2id.get(x, self.src_unk_id) for x in src_sent]
        tgt_ids = [self.tgt_word2id.get(x, self.tgt_unk_id) for x in tgt_sent]

        item = {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "src_sents": src_sent,
            "tgt_sents": tgt_sent,
        }

        return item
    
    def generate_batch(self, item_list):
        src_tgt = [(x["src_ids"], x["tgt_ids"], x["src_sents"], x["tgt_sents"]) for x in item_list]
        src_tgt = sorted(src_tgt, key=lambda x: len(x[0]), reverse=True)

        src_ids = [x[0] for x in src_tgt]
        tgt_ids = [x[1] for x in src_tgt]
        src_sents = [x[2] for x in src_tgt]
        tgt_sents = [x[3] for x in src_tgt]

        src_lengths = [len(x) for x in src_sents]
        tgt_lengths = [len(x) for x in tgt_sents]

        max_src_len = max(len(x) for x in src_ids)
        src_ids = [x + [self.src_pad_id for _ in range(max_src_len - len(x))] for x in src_ids]

        max_tgt_len = max(len(x) for x in tgt_ids)
        tgt_ids = [x + [self.tgt_pad_id for _ in range(max_tgt_len - len(x))] for x in tgt_ids]

        batch = {
            "src_ids": torch.LongTensor(src_ids),
            "tgt_ids": torch.LongTensor(tgt_ids),
            "src_lengths": src_lengths,
            "tgt_lengths": tgt_lengths,
            "src_sents": src_sents,
            "tgt_sents": tgt_sents,
        }

        return batch
    

def test_dataset():
    import os, sys, json
    from data_process import read_corpus
    from torch.utils.data import DataLoader
    os.chdir(sys.path[0])

    src_file_path = "../data/zh_en_data/train.zh"
    src_sp_model_path = "../data/src.model"

    tgt_file_path  ="../data/zh_en_data/train.en"
    tgt_sp_model_path = "../data/tgt.model"

    src_data = read_corpus(src_file_path, "src", src_sp_model_path)
    tgt_data = read_corpus(tgt_file_path, "tgt", tgt_sp_model_path)

    vocab_path = "../data/vocab.json"
    vocab = json.load(open(vocab_path))

    src_word2id = vocab["src_word2id"]
    tgt_word2id = vocab["tgt_word2id"]
    src_pad_id = src_word2id["<pad>"]
    tgt_pad_id = tgt_word2id["<pad>"]
    src_unk_id = src_word2id["<unk>"]
    tgt_unk_id = tgt_word2id["<unk>"]

    dataset = NMTDataset(
        src_data,
        tgt_data,
        src_word2id,
        tgt_word2id,
        src_pad_id,
        tgt_pad_id,
        src_unk_id,
        tgt_unk_id,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=dataset.generate_batch,
    )

    for batch in dataloader:
        print(batch.keys())
    
if __name__ == "__main__":
    test_dataset()



 

