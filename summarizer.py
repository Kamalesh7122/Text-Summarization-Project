import nltk
import heapq
import re

nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Read input text
text = open("dataset.txt", "r", encoding="utf-8").read()

# Preprocessing
text = re.sub(r'\[[0-9]*\]', ' ', text)
text = re.sub(r'\s+', ' ', text)

sentences = sent_tokenize(text)
stop_words = set(stopwords.words("english"))

word_frequencies = {}

for word in word_tokenize(text.lower()):
    if word.isalpha() and word not in stop_words:
        if word not in word_frequencies:
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1

maximum_frequency = max(word_frequencies.values())

for word in word_frequencies:
    word_frequencies[word] = word_frequencies[word] / maximum_frequency

sentence_scores = {}

for sent in sentences:
    for word in word_tokenize(sent.lower()):
        if word in word_frequencies:
            if sent not in sentence_scores:
                sentence_scores[sent] = word_frequencies[word]
            else:
                sentence_scores[sent] += word_frequencies[word]

# Select top sentences
summary_sentences = heapq.nlargest(3, sentence_scores, key=sentence_scores.get)

summary = ' '.join(summary_sentences)

# Save output
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print("Summary Generated Successfully!")
