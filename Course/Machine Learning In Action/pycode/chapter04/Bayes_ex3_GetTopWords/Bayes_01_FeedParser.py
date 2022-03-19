# -*- coding:UTF-8 -*-

# author: Zhang Jiaqi
# datetime:2021/10/7 20:48
# software:PyCharm

import feedparser

ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
print(len(ny['entries']))
print(ny['entries'])
