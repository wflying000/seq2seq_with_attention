import sentencepiece as spm

def read_corpus(
    file_path, 
    source, 
    sp_model_path,
    vocab_size=2500
):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    @param vocab_size (int): number of unique subwords in
        vocabulary when reading and tokenizing
    """
    data = []
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            subword_tokens = sp.encode_as_pieces(line)
            # only append <s> and </s> to the target sentence
            if source == 'tgt':
                subword_tokens = ['<s>'] + subword_tokens + ['</s>']
            data.append(subword_tokens)

    return data


def test():
    import os, sys
    os.chdir(sys.path[0])

    src_file_path = "../data/zh_en_data/train.zh"
    src_sp_model_path = "../data/src.model"

    tgt_file_path  ="../data/zh_en_data/train.en"
    tgt_sp_model_path = "../data/tgt.model"

    src_data = read_corpus(src_file_path, "src", src_sp_model_path)
    tgt_data = read_corpus(tgt_file_path, "tgt", tgt_sp_model_path)

    print(len(src_data), len(tgt_data))


if __name__ == "__main__":
    test()