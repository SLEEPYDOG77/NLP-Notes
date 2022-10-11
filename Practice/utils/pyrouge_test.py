# -*- coding: utf-8 -*-
# @Time    : 2022/4/5 11:26
# @Author  : Zhang Jiaqi
# @File    : pyrouge_test.py
# @Description:

# from pyrouge import Rouge155
# from pprint import pprint
#
# ref_texts = {
#     'A': "Poor nations pressurise developed countries into granting trade subsidies.",
#     'B': "Developed countries should be pressurized. Business exemptions to poor nations.",
#     'C': "World's poor decide to urge developed nations for business concessions."
# }
# summary_text = "Poor nations demand trade subsidies from developed nations."
#
#
# rouge = Rouge155("E:\pyrouge\evaluation\ROUGE-RELEASE-1.5.5")
# # score = rouge.score_summary(summary_text, ref_texts)
# # pprint(score)
# score = rouge.evaluate(str(summary_text), ref_texts)
# pprint(score)
#
# # # rouge.system_dir = ''

#
# from pyrouge import Rouge155
#
# r = Rouge155()
# r.system_dir = '../test/'
# r.model_dir = '../test/'
# r.system_filename_pattern = '(\d+)_candi.txt'
# r.model_filename_pattern = '(\d+)_ref.txt'
#
# output = r.convert_and_evaluate()
# print(output)

from pyrouge import Rouge155

ref_text = "Poor nations pressurise developed countries into granting trade subsidies."
sum_text = "Poor nations demand trade subsidies from developed nations."

rouge = Rouge155("E:\pyrouge\evaluation\ROUGE-RELEASE-1.5.5")
score = rouge.score_summary(ref_text, sum_text)
print(f"Rouge1 Score: {score}")
