# -*- coding: utf-8 -*-
# @Time    : 2022/3/30 20:05
# @Author  : Zhang Jiaqi
# @File    : data_preprocess.py
# @Description:

import pickle
from loguru import logger

from utils.dict_helper import Dict, UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD

src_char = False
tgt_char = False
share = False
report_every = 100000

def makeVocabulary(filename, trun_length, filter_length, vocab, size):
    """
    构造字典
    :param filename: 文件名
    :param trun_length: 截断长度
    :param filter_length: 过滤长度
    :param char:
    :param vocab: 字典
    :param size: 字典size
    :return:
    """
    print("%s: length limit = %d, truncate length = %d" % (filename, filter_length, trun_length))
    max_length = 0
    with open(filename, encoding="utf-8") as f:
        for sent in f.readlines():
            tokens = sent.strip().split()
            print(tokens)
            max_length = max(max_length, len(tokens))
            for word in tokens:
                vocab.add(word)
                print(f"{word}: {vocab.lookup(word)}")
    # print("Max length of %s = %d" % (filename, max_length))
    print(f"size of vocab: {vocab.size}")

    # if size > 0:
    #     originalSize = vocab.size()
    #     vocab = vocab.prune(size)
    #     print("Created dictionary of size %d (pruned from %d)" % (vocab.size(), originalSize))
    return vocab


def makeData(srcFile, tgtFile, srcDicts, tgtDicts, save_srcFile, save_tgtFile, lim=0):
    """

    :param srcFile:
    :param tgtFile:
    :param srcDicts:
    :param tgtDicts:
    :param save_srcFile:
    :param save_tgtFile:
    :param lim:
    :return:
    """
    sizes = 0
    count, empty_ignored, limit_ignored = 0, 0, 0

    logger.info("Processing %s & %s ..." % (srcFile, tgtFile))
    srcF = open(srcFile, encoding='utf8')
    tgtF = open(tgtFile, encoding='utf8')

    srcIdF = open(save_srcFile + '.id', 'w')
    tgtIdF = open(save_tgtFile + '.id', 'w')
    srcStrF = open(save_srcFile + '.str', 'w', encoding='utf8')
    tgtStrF = open(save_tgtFile + '.str', 'w', encoding='utf8')

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()

        # 文档结束
        if sline == "" and tline == "":
            break

        if sline == "" or tline == "":
            print('WARNING: source and target do not have the same number of sentences')
            break

        sline = sline.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            print('WARNING: ignoring an empty line (' + str(count + 1) + ')')
            empty_ignored += 1
            continue

        sline = sline.lower()
        tline = tline.lower()

        srcWords = sline.split()
        tgtWords = tline.split()

        srcIds = srcDicts.convertToIdx(srcWords, UNK_WORD)
        # tgtIds = tgtDicts.convertToIdx(tgtWords, UNK_WORD, BOS_WORD, EOS_WORD)
        tgtIds = tgtDicts.convertToIdx(tgtWords, UNK_WORD)

        srcIdF.write(" ".join(list(map(str, srcIds))) + "\n")
        tgtIdF.write(" ".join(list(map(str, tgtIds))) + "\n")

        srcStrF.write(" ".join(srcWords) + '\n')
        tgtStrF.write(" ".join(tgtWords) + '\n')

        sizes += 1
        count += 1

    srcF.close()
    tgtF.close()
    srcStrF.close()
    tgtStrF.close()
    srcIdF.close()
    tgtIdF.close()

    print('Prepared %d sentences (%d and %d ignored due to length == 0 or > )' %
          (sizes, empty_ignored, limit_ignored))

    return {
        'srcF': save_srcFile + '.id',
        'tgtF': save_tgtFile + '.id',
        'original_srcF': save_srcFile + '.str',
        'original_tgtF': save_tgtFile + '.str',
        'length': sizes
    }


def main():
    dicts = {}

    # load_data = '../0data/lcsts_data/data_prepared_test/'
    # save_data = '../0data/lcsts_data/data_preprocessed_test/'
    load_data = '../0data/gigaword_data/data_prepared_test/'
    save_data = '../0data/gigaword_data/data_preprocessed_test/'

    train_src, train_tgt = load_data + 'train.src', load_data + 'train.tgt'
    valid_src, valid_tgt = load_data + 'valid.src', load_data + 'valid.tgt'
    test_src, test_tgt = load_data + 'test.src', load_data + 'test.tgt'

    save_train_src, save_train_tgt = save_data + 'train.src', save_data + 'train.tgt'
    save_valid_src, save_valid_tgt = save_data + 'valid.src', save_data + 'valid.tgt'
    save_test_src, save_test_tgt = save_data + 'test.src', save_data + 'test.tgt'

    src_dict, tgt_dict = save_data + 'src.dict', save_data + 'tgt.dict'

    src_vocab_size = 50000
    tgt_vocab_size = 50000

    logger.info("Building source and target vocabulary...")

    dicts['src'] = Dict([PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD])
    dicts['src'] = makeVocabulary(filename=train_src, trun_length=0, filter_length=0, vocab=dicts['src'], size=src_vocab_size)

    dicts['tgt'] = Dict([PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD])
    dicts['tgt'] = makeVocabulary(filename=train_tgt, trun_length=0, filter_length=0, vocab=dicts['tgt'], size=tgt_vocab_size)

    logger.info("Preparing training...")
    train = makeData(train_src, train_tgt, dicts['src'], dicts['tgt'], save_train_src, save_train_tgt)

    logger.info("Preparing validation...")
    valid = makeData(valid_src, valid_tgt, dicts['src'], dicts['tgt'], save_valid_src, save_valid_tgt)

    logger.info("Preparing testing...")
    test = makeData(test_src, test_tgt, dicts['src'], dicts['tgt'], save_test_src, save_test_tgt)

    print('Saving source vocabulary to \'' + src_dict + '\'...')
    dicts['src'].writeFile(src_dict)

    print('Saving source vocabulary to \'' + tgt_dict + '\'...')
    dicts['tgt'].writeFile(tgt_dict)

    data = {
        'train': train,
        'valid': valid,
        'test': test,
        'dict': dicts
    }
    pickle.dump(data, open(save_data + '0data.pkl', 'wb'))

if __name__ == "__main__":
    main()
