# -*- coding: utf-8 -*-
# @Time    : 2022/3/9 20:57
# @Author  : Zhang Jiaqi
# @File    : from_tensor_slices_test.py
# @Description:

import tensorflow as tf
import numpy as np

features, labels = (np.random.sample((6, 3)),
                    np.random.sample((6, 1)))

print((features, labels))
dataset = tf.data.Dataset.from_tensor_slices((features, labels))


