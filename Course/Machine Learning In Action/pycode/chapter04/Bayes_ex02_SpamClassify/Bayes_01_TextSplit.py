# -*- coding:UTF-8 -*-

# author: Zhang Jiaqi
# datetime:2021/10/7 20:22
# software:PyCharm
import re

mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
# print(mySent.split())

regEx = re.compile('\W+')
# listOfTokens = regEx.split(mySent)
# print(listOfTokens)

# print([tok for tok in listOfTokens if len(tok) > 0])

# 将字符串全部转换成小写
# print([tok.lower() for tok in listOfTokens if len(tok) > 0])


emailText = open('email/ham/6.txt').read()
listOfTokens = regEx.split(emailText)
print(listOfTokens)
