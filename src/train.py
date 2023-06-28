import os
import sys
import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader

from model import NMT
from vocab import Vocab
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
    eval_src_data = read_corpus(eval_src_file_path, 'src', src_sp_model_path)
    eval_tgt_data = read_corpus(eval_tgt_file_path, 'tgt', tgt_sp_model_path)

    vocab_path = "../data/vocab.json"
    vocab_data = json.load(open(vocab_path))
    vocab = Vocab.load(vocab_path)

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
        # num_workers=8,
        # pin_memory=True,
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
        # num_workers=8,
        # pin_memory=True,
    )

    hidden_size = 768
    embed_size = 1024
    dropout_rate = 0.3
    cuda = True
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    uniform_init = 0.1
    learning_rate = 5e-4

    model_config = ModelConfig(
        hidden_size=hidden_size,
        embed_size=embed_size,
        src_vocab_size=len(src_word2id),
        tgt_vocab_size=len(tgt_word2id),
        src_pad_id=src_pad_id,
        tgt_pad_id=tgt_pad_id,
        dropout_rate=dropout_rate,
        vocab=vocab,
    )

    model = NMT(model_config)
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)
    
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 30
    clip_grad = 5.0
    log_step = 10
    eval_step = 200
    patience = 1
    lr_decay = 0.5
    max_num_trial = 5
    now_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    outputs_dir = f"../outputs/{now_time}/"
    os.makedirs(outputs_dir)

    training_args = TrainingArguments(
        num_epochs=num_epochs,
        clip_grad=clip_grad,
        log_step=log_step,
        eval_step=eval_step,
        patience=patience,
        lr_decay=lr_decay,
        max_num_trial=max_num_trial,
    )

    trainer = Trainer(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        training_args,
        outputs_dir,
    )

    trainer.train()


if __name__ == "__main__":
    os.chdir(sys.path[0])
    train()



