from collections import namedtuple

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

ModelConfig = namedtuple(
    "ModelConfig", 
    ["hidden_size", "embed_size", "src_vocab_size", "tgt_vocab_size",
     "src_pad_id", "tgt_pad_id", ]
)