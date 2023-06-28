import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from configuration import Hypothesis


class NMT(nn.Module):
    def __init__(self, config):
        super(NMT, self).__init__()
        self.embed_size = config.embed_size
        self.hidden_size = config.hidden_size
        embed_size = config.embed_size
        hidden_size = config.hidden_size

        self.src_embedding = nn.Embedding(
            num_embeddings=config.src_vocab_size, 
            embedding_dim=config.embed_size, 
            padding_idx=config.src_pad_id,
        )
        self.tgt_embedding = nn.Embedding(
            num_embeddings=config.tgt_vocab_size,
            embedding_dim=config.embed_size,
            padding_idx=config.tgt_pad_id,
        )

        self.post_embed_cnn = nn.Conv1d(
            in_channels=config.embed_size, 
            out_channels=config.embed_size,
            kernel_size=2,
            padding="same",
        )
        self.encoder = nn.LSTM(
            input_size=config.embed_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            bias=True,
            batch_first=False,
            bidirectional=True,
        )
        self.decoder = nn.LSTMCell(
            input_size=embed_size + hidden_size,
            hidden_size=hidden_size,
            bias=True,
        )
        self.h_projection = nn.Linear(hidden_size*2, hidden_size, bias=False)
        self.c_projection = nn.Linear(hidden_size*2, hidden_size, bias=False)
        self.att_projection = nn.Linear(hidden_size*2, hidden_size, bias=False)
        self.combined_output_projection = nn.Linear(hidden_size*3, hidden_size, bias=False)
        self.target_vocab_projection = nn.Linear(hidden_size, config.tgt_vocab_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.dropout_rate = config.dropout_rate
        self.vocab = config.vocab

        self.tgt_pad_id = config.tgt_pad_id
        self.config = config

    
    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor, src_lengths: List[int]) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param src_ids: list of source sentence tokens, shape: (src_len, b)
        @param tgt_ids: list of target sentence tokens, wrapped by `<s>` and `</s>`, shape: (tgt_len, b)

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        

        enc_hiddens, dec_init_state = self.encode(src_ids, src_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, src_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, tgt_ids)
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (tgt_ids != self.tgt_pad_id).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=tgt_ids[1:].unsqueeze(-1), dim=-1).squeeze(
            -1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores


    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.

        @param source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b), where
                                        b = batch_size, src_len = maximum source sentence length. Note that
                                       these have already been sorted in order of longest to shortest sentence.
        @param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
        @returns enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h*2), where
                                        b = batch size, src_len = maximum source sentence length, h = hidden size.
        @returns dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's initial
                                                hidden state and cell. Both tensors should have shape (2, b, h).
        """
        enc_hiddens, dec_init_state = None, None

        # 取src的embedding并做处理获得src的embedding表示X
        source_padded_t = source_padded.transpose(0, 1) # (b, src_len)
        X = self.src_embedding(source_padded_t) # (b, src_len, e)
        X = X.transpose(0, 1) # (src_len, b, e)
        X = self.post_embed_cnn(torch.permute(X, (1, 2, 0))) # X: (b, e, src_len)
        X = torch.permute(X, (2, 0, 1)) # (src_len, b, e)
        
        # 通过lstm对X进行处理
        X = pack_padded_sequence(input=X, lengths=source_lengths)
        enc_hiddens, (last_hidden, last_cell) = self.encoder(X) # (src_len, b, h*2), (2, b, h), (2, b, h)
        enc_hiddens, lens_unpacked = pad_packed_sequence(enc_hiddens)
        enc_hiddens = torch.permute(enc_hiddens, (1, 0, 2)) # (b, src_len, h*2)

        # 将正向和反向的h拼接并做变换后作为decoder的初始h
        enc_hidden_cat = torch.cat([last_hidden[0], last_hidden[1]], dim=1)
        init_decoder_hidden = self.h_projection(enc_hidden_cat)

        # 将正向和法相的c拼接并做变换后作为decoder的初始c
        enc_cell_cat = torch.cat([last_cell[0], last_cell[1]], dim=1)
        init_decoder_cell = self.c_projection(enc_cell_cat)

        dec_init_state = (init_decoder_hidden, init_decoder_cell)

        return enc_hiddens, dec_init_state

    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
               dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:
        """Compute combined output vectors for a batch.

        @param enc_hiddens (Tensor): Hidden states (b, src_len, h*2), where
                                     b = batch size, src_len = maximum source sentence length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks (b, src_len), where
                                     b = batch size, src_len = maximum source sentence length.
        @param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
        @param target_padded (Tensor): Gold-standard padded target sentences (tgt_len, b), where
                                       tgt_len = maximum target sentence length, b = batch size.

        @returns combined_outputs (Tensor): combined output tensor  (tgt_len, b,  h), where
                                        tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        # Chop off the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        enc_hiddens_proj = self.att_projection(enc_hiddens) # (b, src_len, h)
        target_padded_t = target_padded.transpose(0, 1) # (b, tgt_len)
        Y = self.tgt_embedding(target_padded_t) # (b, tgt_len, e)
        Y = Y.transpose(0, 1) # (tgt_len, b, e)

        for Y_t in torch.split(Y, split_size_or_sections=1, dim=0):
            Y_t = Y_t.squeeze(0) # (b, e)
            Ybar_t = torch.cat([o_prev, Y_t], dim=1) # (b, h + e)
            dec_state, combined_output, e_t = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            combined_outputs.append(combined_output)
            o_prev = combined_output

        combined_outputs = torch.stack(combined_outputs, dim=0) # (tgt_len, b, h)

        return combined_outputs
    
    def step(self, Ybar_t: torch.Tensor,
             dec_state: Tuple[torch.Tensor, torch.Tensor],
             enc_hiddens: torch.Tensor,
             enc_hiddens_proj: torch.Tensor,
             enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length.

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """

        combined_output = None

        dec_state = self.decoder(Ybar_t, dec_state)
        dec_hidden, dec_cell = dec_state # (b, h), (b, h)
        e_t = torch.bmm(dec_hidden.unsqueeze(1), enc_hiddens_proj.transpose(1, 2)) # (b, 1, src_len)
        e_t = e_t.squeeze(1) # (b, src_len)

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        alpha_t = torch.softmax(e_t, dim=-1)
        a_t = torch.bmm(alpha_t.unsqueeze(1), enc_hiddens) # (b, 1, 2h)
        a_t = a_t.squeeze(1) # (b, 2h)
        U_t = torch.cat([a_t, dec_hidden], dim=1) # (b, 3h)
        V_t = self.combined_output_projection(U_t) # (b, h)
        O_t = self.dropout(torch.tanh(V_t)) # (b, h)

        combined_output = O_t
        return dec_state, combined_output, e_t
    
    def generate_sent_masks(self, enc_hiddens: torch.Tensor, src_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size.
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.

        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(src_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    def beam_search(self, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70) -> List[
        Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.tgt_embedding(y_tm1)

            # x = torch.cat([y_t_embed, att_tm1], dim=-1)
            x = torch.cat([att_tm1, y_t_embed], dim=-1)
            (h_t, cell_t), att_t, _ = self.step(x, h_tm1,
                                                exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1) # 求score应该用概率相乘，这里求和是因为log_p_t是log(p), log(p)相加等价于概率相乘
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = torch.div(top_cand_hyp_pos, len(self.vocab.tgt), rounding_mode='floor') # 由于上一步求最高的概率是将所有概率混起来求，这里是解开最高的k个值分别属于哪个解码的序列
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses
    
    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.src_embedding.weight.device
    
    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location="cpu")
        config = params["config"]
        model = NMT(config)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'config': self.config,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)