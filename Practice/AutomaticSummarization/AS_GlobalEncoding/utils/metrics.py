# -*- coding: utf-8 -*-
# @Time    : 2022/4/5 10:31
# @Author  : Zhang Jiaqi
# @File    : metrics.py
# @Description:

# import pyrouge
import codecs
import os
from rouge import FilesRouge
from rouge import Rouge
from pprint import pprint
import json

def bleu(reference, candidate, log_path, print_log, config):
    ref_file = log_path+'reference.txt'
    cand_file = log_path+'candidate.txt'
    with codecs.open(ref_file, 'w', 'utf-8') as f:
        for s in reference:
            if not config.char:
                f.write(" ".join(s)+'\n')
            else:
                f.write("".join(s) + '\n')
    with codecs.open(cand_file, 'w', 'utf-8') as f:
        for s in candidate:
            if not config.char:
                f.write(" ".join(s).strip()+'\n')
            else:
                f.write("".join(s).strip() + '\n')

    if config.refF != '':
        ref_file = config.refF

    temp = log_path + "result.txt"
    command = "perl script/multi-bleu.perl " + ref_file + "<" + cand_file + "> " + temp
    os.system(command)
    with open(temp) as ft:
        result = ft.read()
    os.remove(temp)
    print_log(result)

    return float(result.split()[2][:-1])

def rouge(reference, candidate, log_path, print_log, config):
    assert len(reference) == len(candidate)

    ref_dir = log_path + 'reference/'
    cand_dir = log_path + 'candidate/'
    if not os.path.exists(ref_dir):
        os.mkdir(ref_dir)
    if not os.path.exists(cand_dir):
        os.mkdir(cand_dir)

    hyp_path = log_path + 'candidate.txt'
    ref_path = log_path + 'reference.txt'
    print(f'hyp_path: {hyp_path}')
    print(f'ref_path: {ref_path}\n')

    for i in range(len(reference)):
        with codecs.open(ref_dir + "%06d_reference.txt" % i, 'w', 'utf-8') as f:
            f.write(" ".join(reference[i]).replace(' <\s>', '\n').replace('<s>', '') + '\n')
        with codecs.open(cand_dir + "%06d_candidate.txt" % i, 'w', 'utf-8') as f:
            f.write(" ".join(candidate[i]).replace(' <\s>', '\n').replace('<s>', '').replace('<unk>', 'UNK') + '\n')

    scores_all = {
        "recall": {
            "rouge-1": 0.0,
            "rouge-2": 0.0,
            "rouge-l": 0.0,
        },
        "precision": {
            "rouge-1": 0.0,
            "rouge-2": 0.0,
            "rouge-l": 0.0,
        },
        "f_score": {
            "rouge-1": 0.0,
            "rouge-2": 0.0,
            "rouge-l": 0.0,
        }
    }


    for i in range(len(reference)):
        with codecs.open(ref_dir + "%06d_reference.txt" % i, 'r', 'utf-8') as f:
            reference = f.readline()
            # print(f"reference: {reference}")
        with codecs.open(cand_dir + "%06d_candidate.txt" % i, 'r', 'utf-8') as f:
            candidate = f.readline()
            # print(f"candidate: {candidate}")
        rouge = Rouge()
        score = rouge.get_scores(candidate, reference)
        score = score[0]
        scores_all['recall']['rouge-1'] += round(score["rouge-1"]["r"] * 100, 2)
        scores_all['recall']['rouge-2'] += round(score["rouge-2"]["r"] * 100, 2)
        scores_all['recall']['rouge-l'] += round(score["rouge-l"]["r"] * 100, 2)
        scores_all['precision']['rouge-1'] += round(score["rouge-1"]["p"] * 100, 2)
        scores_all['precision']['rouge-2'] += round(score["rouge-2"]["p"] * 100, 2)
        scores_all['precision']['rouge-l'] += round(score["rouge-l"]["p"] * 100, 2)
        scores_all['f_score']['rouge-1'] += round(score["rouge-1"]["f"] * 100, 2)
        scores_all['f_score']['rouge-2'] += round(score["rouge-2"]["f"] * 100, 2)
        scores_all['f_score']['rouge-l'] += round(score["rouge-l"]["f"] * 100, 2)

    scores_all['recall']['rouge-1'] /= len(reference)
    scores_all['recall']['rouge-2'] /= len(reference)
    scores_all['recall']['rouge-l'] /= len(reference)
    scores_all['precision']['rouge-1'] /= len(reference)
    scores_all['precision']['rouge-2'] /= len(reference)
    scores_all['precision']['rouge-l'] /= len(reference)
    scores_all['f_score']['rouge-1'] /= len(reference)
    scores_all['f_score']['rouge-2'] /= len(reference)
    scores_all['f_score']['rouge-l'] /= len(reference)

    pprint(scores_all)

    return scores_all['f_score'], scores_all['recall'], scores_all['precision']

    # files_rouge = FilesRouge()
    # scores = files_rouge.get_scores(hyp_path, ref_path, avg=True, ignore_empty=True)
    #
    # recall = [round(scores["rouge-1"]["r"] * 100, 2),
    #           round(scores["rouge-2"]["r"] * 100, 2),
    #           round(scores["rouge-l"]["r"] * 100, 2)]
    # precision = [round(scores["rouge-1"]["p"] * 100, 2),
    #              round(scores["rouge-2"]["p"] * 100, 2),
    #              round(scores["rouge-l"]["p"] * 100, 2)]
    # f_score = [round(scores["rouge-1"]["f"] * 100, 2),
    #            round(scores["rouge-2"]["f"] * 100, 2),
    #            round(scores["rouge-l"]["f"] * 100, 2)]
    # print_log("F_measure: %s Recall: %s Precision: %s\n"
    #           % (str(f_score), str(recall), str(precision)))
    #
    # return f_score[:], recall[:], precision[:]

    # r = pyrouge.Rouge155("E:\pyrouge\evaluation\ROUGE-RELEASE-1.5.5")
    # r.model_filename_pattern = '#ID#_reference.txt'
    # r.system_filename_pattern = '(\d+)_candidate.txt'
    # r.model_dir = ref_dir
    # r.system_dir = cand_dir
    # logging.getLogger('global').setLevel(logging.WARNING)
    # rouge_results = r.convert_and_evaluate()
    # scores = r.output_to_dict(rouge_results)

    # file_rouge = FilesRouge()
    # scores = file_rouge.get_scores(cand_dir, ref_dir, ignore_empty=True, avg=True)
    # recall = [round(scores["rouge-1"]["r"] * 100, 2),
    #           round(scores["rouge-2"]["r"] * 100, 2),
    #           round(scores["rouge-l"]["r"] * 100, 2)]
    # precision = [round(scores["rouge-1"]["p"] * 100, 2),
    #              round(scores["rouge-2"]["p"] * 100, 2),
    #              round(scores["rouge-l"]["p"] * 100, 2)]
    # f_score = [round(scores["rouge-1"]["f"] * 100, 2),
    #            round(scores["rouge-2"]["f"] * 100, 2),
    #            round(scores["rouge-l"]["f"] * 100, 2)]
    # print_log("F_measure: %s Recall: %s Precision: %s\n"
    #           % (str(f_score), str(recall), str(precision)))
    #
    # return f_score[:], recall[:], precision[:]


# def rouge(reference, candidate, log_path, print_log, config):
#     assert len(reference) == len(candidate)
#
#     ref_dir = log_path + 'reference/'
#     cand_dir = log_path + 'candidate/'
#     if not os.path.exists(ref_dir):
#         os.mkdir(ref_dir)
#     if not os.path.exists(cand_dir):
#         os.mkdir(cand_dir)
#
#     for i in range(len(reference)):
#         with codecs.open(ref_dir+"%06d_reference.txt" % i, 'w', 'utf-8') as f:
#             f.write(" ".join(reference[i]).replace(' <\s> ', '\n') + '\n')
#         with codecs.open(cand_dir+"%06d_candidate.txt" % i, 'w', 'utf-8') as f:
#             f.write(" ".join(candidate[i]).replace(' <\s> ', '\n').replace('<unk>', 'UNK') + '\n')
#
#     r = pyrouge.Rouge155("E:\pyrouge\evaluation\ROUGE-RELEASE-1.5.5")
#     r.model_filename_pattern = '#ID#_reference.txt'
#     r.system_filename_pattern = '(\d+)_candidate.txt'
#     r.model_dir = ref_dir
#     r.system_dir = cand_dir
#     logging.getLogger('global').setLevel(logging.WARNING)
#     rouge_results = r.convert_and_evaluate()
#     scores = r.output_to_dict(rouge_results)
#     recall = [round(scores["rouge_1_recall"] * 100, 2),
#               round(scores["rouge_2_recall"] * 100, 2),
#               round(scores["rouge_l_recall"] * 100, 2)]
#     precision = [round(scores["rouge_1_precision"] * 100, 2),
#                  round(scores["rouge_2_precision"] * 100, 2),
#                  round(scores["rouge_l_precision"] * 100, 2)]
#     f_score = [round(scores["rouge_1_f_score"] * 100, 2),
#                round(scores["rouge_2_f_score"] * 100, 2),
#                round(scores["rouge_l_f_score"] * 100, 2)]
#     print_log("F_measure: %s Recall: %s Precision: %s\n"
#               % (str(f_score), str(recall), str(precision)))
#
#     return f_score[:], recall[:], precision[:]

