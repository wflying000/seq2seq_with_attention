import json
import torch
from torch.utils.data import DataLoader

from model import NMT
from trainer import Trainer
from dataset import NMTDataset
from data_process import read_corpus
from configuration import ModelConfig, TrainingArguments


def train():
    
    # 获取训练数据
    train_src_file_path = "../data/zh_en_data/train.zh"
    src_sp_model_path = "../data/src.model"
    train_tgt_file_path  ="../data/zh_en_data/train.en"
    tgt_sp_model_path = "../data/tgt.model"
    train_src_data = read_corpus(train_src_file_path, "src", src_sp_model_path)
    train_tgt_data = read_corpus(train_tgt_file_path, "tgt", tgt_sp_model_path)

    # 获取验证数据
    eval_src_file_path = "../data/zh_en_data/dev.zh"
    eval_tgt_file_path = "../data/zh_en_data/dev.en"
    eval_src_data = read_corpus(eval_src_file_path, source='src', vocab_size=3000)
    eval_tgt_data = read_corpus(eval_tgt_file_path, source='tgt', vocab_size=2000)

    vocab_path = "../data/vocab.json"
    vocab_data = json.load(open(vocab_path))

    src_word2id = vocab_data["src_word2id"]
    tgt_word2id = vocab_data["tgt_word2id"]
    src_pad_id = src_word2id["<pad>"]
    tgt_pad_id = tgt_word2id["<pad>"]
    src_unk_id = src_word2id["<unk>"]
    tgt_unk_id = tgt_word2id["<unk>"]

    batch_size = 32

    train_dataset = NMTDataset(
        src_data=train_src_data,
        tgt_data=train_tgt_data,
        src_word2id=src_word2id,
        tgt_word2id=tgt_word2id,
        src_pad_id=src_pad_id,
        tgt_pad_id=tgt_pad_id,
        src_unk_id=src_unk_id,
        tgt_unk_id=tgt_unk_id,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.generate_batch,
        num_workers=8,
        pin_memory=True,
    )

    eval_dataset = NMTDataset(
        src_data=eval_src_data,
        tgt_data=eval_tgt_data,
        src_word2id=src_word2id,
        tgt_word2id=tgt_word2id,
        src_pad_id=src_pad_id,
        tgt_pad_id=tgt_pad_id,
        src_unk_id=src_unk_id,
        tgt_unk_id=tgt_unk_id,
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=eval_dataset.generate_batch,
        num_workers=8,
        pin_memory=True,
    )



