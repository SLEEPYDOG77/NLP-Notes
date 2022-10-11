# -*- coding: utf-8 -*-
# @Time    : 2022/4/9 11:24
# @Author  : Zhang Jiaqi
# @File    : log_operate.py
# @Description: 通用日志工具

import json


class LogOperator(object):
    def __init__(self, logpath):
        self.logpath = logpath

    def get_logpath(self):
        return self.logpath

    def write_model(self, params):
        with open(self.logpath, 'a', encoding='utf-8') as f:
            json.dump(params, f, indent=4)
            f.write("\n")
        f.close()

    def write_log(self, log):
        with open(self.logpath, 'a', encoding='utf-8') as f:
            f.write(log + "\n\n")
        f.close()



