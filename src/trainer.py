import os
import sys
import time
import math
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer():
    def __init__(
        self,
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        training_args,
        outputs_dir,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.training_args = training_args
        self.outputs_dir = outputs_dir
        self.writer = SummaryWriter(outputs_dir)

    def train(self):
        model = self.model
        device = model.device
        args = self.training_args
        num_train_samples = len(self.train_dataloader)

        num_trial = 0
        patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
        cum_examples = report_examples = epoch = valid_num = 0
        train_time = begin_time = time.time()
        hist_eval_scores = []

        for epoch in tqdm(range(args.num_epochs), total=args.num_epochs):
            model.train()

            for idx, batch in tqdm(enumerate(self.train_dataloader), total=num_train_samples, leave=False):
                src_ids = batch["src_ids"].to(device)
                tgt_ids = batch["tgt_ids"].to(device)
                src_ids = src_ids.transpose(0, 1)
                tgt_ids = tgt_ids.transpose(0, 1)
                src_lengths = batch["src_lengths"]
                tgt_lengths = batch["tgt_lengths"]

                batch_size = len(src_lengths)

                example_losses = -model(src_ids, tgt_ids, src_lengths)

                batch_loss = example_losses.sum()
                loss = batch_loss / batch_size

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                self.optimizer.step()
                
                batch_losses_val = batch_loss.item()
                report_loss += batch_losses_val
                cum_loss += batch_losses_val

                num_tgt_words_to_predict = sum(tgt_lengths) - batch_size # 每个样本去掉起始符号<s>
                report_tgt_words += num_tgt_words_to_predict
                cum_tgt_words += num_tgt_words_to_predict
                report_examples += batch_size
                cum_examples += batch_size

                global_step = num_train_samples * epoch + idx

                # 打印训练集结果
                if global_step % args.log_step == 0:
                    self.writer.add_scalar("loss/train", report_loss / report_examples, global_step)
                    self.writer.add_scalar("perplexity/train", math.exp(report_loss / report_tgt_words), global_step)
                    print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                        'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, global_step,
                                                                                            report_loss / report_examples,
                                                                                            math.exp(report_loss / report_tgt_words),
                                                                                            cum_examples,
                                                                                            report_tgt_words / (time.time() - train_time),
                                                                                            time.time() - begin_time), file=sys.stderr)

                    train_time = time.time()
                    report_loss = report_tgt_words = report_examples = 0.
                

                if global_step % args.eval_step == 0:
                    self.writer.add_scalar("loss/val", cum_loss / cum_examples, global_step)
                    print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, global_step,
                                                                                            cum_loss / cum_examples,
                                                                                            np.exp(cum_loss / cum_tgt_words),
                                                                                            cum_examples), file=sys.stderr)

                    cum_loss = cum_examples = cum_tgt_words = 0.
                    valid_num += 1

                    eval_ppl = self.evaluate_ppl()   # dev batch size can be a bit larger
                    eval_metric = -eval_ppl

                    self.writer.add_scalar("perplexity/eval", eval_ppl, global_step)
                    print('eval: step %d, dev. ppl %f' % (global_step, eval_ppl), file=sys.stderr)

                    is_better = len(hist_eval_scores) == 0 or eval_metric > max(hist_eval_scores)
                    hist_eval_scores.append(eval_metric)

                    if is_better:
                        patience = 0
                        model_save_name = f"model_step_{global_step}_ppl_{eval_metric}.pth"
                        model_save_path = os.path.join(self.outputs_dir, model_save_name)
                        print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                        model.save(model_save_path)

                        # also save the optimizers' state
                        torch.save(self.optimizer.state_dict(), model_save_path + '.optim')
                        
                    elif patience < int(args.patience):
                        patience += 1
                        print('hit patience %d' % patience, file=sys.stderr)

                        if patience == int(args.patience):
                            num_trial += 1
                            print('hit #%d trial' % num_trial, file=sys.stderr)
                            if num_trial == int(args.max_num_trial):
                                print('early stop!', file=sys.stderr)
                                exit(0)

                            # decay lr, and restore from previously best checkpoint
                            lr = self.optimizer.param_groups[0]['lr'] * float(args.lr_decay)
                            print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                            # load model
                            params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                            model.load_state_dict(params['state_dict'])
                            model = model.to(device)

                            print('restore parameters of the optimizers', file=sys.stderr)
                            self.optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                            # set new lr
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = lr

                            # reset patience
                            patience = 0



    def evaluate_ppl(self):
        """ Evaluate perplexity on dev sentences
        @param model (NMT): NMT Model
        @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
        @param batch_size (batch size)
        @returns ppl (perplixty on dev sentences)
        """
        was_training = self.model.training
        self.model.eval()
        device = self.model.device
        cum_loss = 0.
        cum_tgt_words = 0.

        # no_grad() signals backend to throw away all gradients
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, total=len(self.eval_dataloader), leave=False, desc="eval"):
                src_ids = batch["src_ids"].to(device)
                tgt_ids = batch["tgt_ids"].to(device)
                src_ids = src_ids.transpose(0, 1)
                tgt_ids = tgt_ids.transpose(0, 1)
                src_lengths = batch["src_lengths"]
                tgt_lengths = batch["tgt_lengths"]

                batch_size = len(src_lengths)

                loss = -self.model(src_ids, tgt_ids, src_lengths).sum()

                cum_loss += loss.item()
                tgt_word_num_to_predict = sum(tgt_lengths) - batch_size  # 每个样本去掉起始符号<s>
                cum_tgt_words += tgt_word_num_to_predict

            ppl = np.exp(cum_loss / cum_tgt_words)

        if was_training:
            self.model.train()

        return ppl

