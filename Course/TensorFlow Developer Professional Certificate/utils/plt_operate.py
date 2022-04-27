# -*- coding: utf-8 -*-
# @Time    : 2022/4/27 14:39
# @Author  : Zhang Jiaqi
# @File    : plt_operate.py
# @Description:
import os.path

from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class PltOperate(object):
    def __init__(self):
        logger.info("mysql_operate ---> init")

    def show_source_imgs(self, ncols, nrows, pic_index=0):
        # fig = plt.gcf()
        # fig.get_size_inches(ncols * 4, nrows * 4)
        #
        # pic_index += 8
        # next_cat_pix = [os.path.join(tra)]
        # TODO:
        pass


