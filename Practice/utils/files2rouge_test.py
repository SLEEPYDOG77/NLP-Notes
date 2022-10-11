# -*- coding: utf-8 -*-
# @Time    : 2022/4/5 14:46
# @Author  : Zhang Jiaqi
# @File    : files2rouge_test.py
# @Description:

import files2rouge

result = files2rouge.run('../Global Encoding/seq2seq_20220408/giga/1649140423769/candidate.txt',
                         '../Global Encoding/seq2seq_20220408/giga/1649140423769/reference.txt',
                         ignore_empty_summary=True)
print(result)
