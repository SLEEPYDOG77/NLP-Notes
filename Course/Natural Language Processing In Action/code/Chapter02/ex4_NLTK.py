from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TreebankWordTokenizer

def tokenization_1(sentence):
    tokenizer = RegexpTokenizer(r'\w+|$[0-9.]+|\S+')
    print(tokenizer.tokenize(sentence))

def tokenization_2(sentence):
    tokenizer = TreebankWordTokenizer()
    print(tokenizer.tokenize(sentence))


if __name__ == "__main__":
    sentence = """Thomas Jefferson began building Monticello at the age of 26."""

    # tokenization_1(sentence=sentence)
    tokenization_2(sentence=sentence)
