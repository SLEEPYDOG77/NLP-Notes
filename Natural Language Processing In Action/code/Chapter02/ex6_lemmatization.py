import nltk

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("better")
lemmatizer.lemmatize("better", pos="a") # adj
lemmatizer.lemmatize("good", pos="a")   # adj
lemmatizer.lemmatize("goods", pos="n")  # n
lemmatizer.lemmatize("goodness", pos="n")   # n
lemmatizer.lemmatize("best", pos="a")   # adj
