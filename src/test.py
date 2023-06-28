import os
import sys
import json
import torch
import jsonlines
import sacrebleu
from tqdm import tqdm
from typing import List

from model import NMT
from data_process import read_corpus
from configuration import Hypothesis, ModelConfig

def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps)

    if was_training: model.train(was_training)

    return hypotheses

def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    # remove the start and end tokens
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    
    # detokenize the subword pieces to get full sentences
    detokened_refs = [''.join(pieces).replace('▁', ' ') for pieces in references]
    detokened_hyps = [''.join(hyp.value).replace('▁', ' ') for hyp in hypotheses]

    # sacreBLEU can take multiple references (golden example per sentence) but we only feed it one
    bleu = sacrebleu.corpus_bleu(detokened_hyps, [detokened_refs])

    return bleu.score

def test():
    test_src_filepath = "../data/zh_en_data/test.zh"
    test_tgt_filepath = "../data/zh_en_data/test.en"
    src_sp_model_path = "../data/src.model"
    tgt_sp_model_path = "../data/tgt.model"
    max_decoding_time_step = 70
    
    test_src_data = read_corpus(test_src_filepath, "src", src_sp_model_path)
    test_tgt_data = read_corpus(test_tgt_filepath, "tgt", tgt_sp_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "../outputs/20230628-214338/model_step_18600_ppl_-11.551021947266559.pth"
    model = NMT.load(model_path)
    model = model.to(device)

    hypotheses = beam_search(model, test_src_data,
                             beam_size=10,
                             max_decoding_time_step=max_decoding_time_step)

    
    top_hypotheses = [hyps[0] for hyps in hypotheses]
    bleu_score = compute_corpus_level_bleu_score(test_tgt_data, top_hypotheses)
    print('Corpus BLEU: {}'.format(bleu_score), file=sys.stderr)

    test_outputs_filepath = "../outputs/test_outputs.jsonl"

    with jsonlines.open(test_outputs_filepath, 'w') as f:
        for src_sent, hyps in zip(test_src_data, hypotheses):
            src_sent = ''.join(src_sent).replace("▁", "")
            top_hyp = hyps[0]
            hyp_sent = ''.join(top_hyp.value).replace('▁', ' ')
            f.write({"src": src_sent, "tgt": hyp_sent})
        


if __name__ == "__main__":
    os.chdir(sys.path[0])
    test()