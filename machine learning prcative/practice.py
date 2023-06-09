import spacy
from spacy.tokens import DocBin
import pandas as pd
from sklearn.model_selection import train_test_split
import csv

with open('Stess_data.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    data = list(reader)

# Split data into training and validation sets
train_data, valid_data = train_test_split(data, test_size=0.2)
print(train_data[3])

nlp = spacy.load("en_core_web_sm")

def make_docs(data):
    docs = []
    for doc, label in nlp.pipe(data, as_tuples=True):
        doc.cats = {}
        if label == "0":
            doc.cats["Sadness"] = 1
            doc.cats["Joy"] = 0
            doc.cats["Love"] = 0
            doc.cats["Anger"] = 0
            doc.cats["Fear"] = 0
            doc.cats["Surprise"] = 0
        elif label == "1":
            doc.cats["Sadness"] = 0
            doc.cats["Joy"] = 1
            doc.cats["Love"] = 0
            doc.cats["Anger"] = 0
            doc.cats["Fear"] = 0
            doc.cats["Surprise"] = 0
        elif label == "2":
            doc.cats["Sadness"] = 0
            doc.cats["Joy"] = 0
            doc.cats["Love"] = 1
            doc.cats["Anger"] = 0
            doc.cats["Fear"] = 0
            doc.cats["Surprise"] = 0
        elif label == "3":
            doc.cats["Sadness"] = 0
            doc.cats["Joy"] = 0
            doc.cats["Love"] = 0
            doc.cats["Anger"] = 1
            doc.cats["Fear"] = 0
            doc.cats["Surprise"] = 0
        elif label == "4":
            doc.cats["Sadness"] = 0
            doc.cats["Joy"] = 0
            doc.cats["Love"] = 0
            doc.cats["Anger"] = 0
            doc.cats["Fear"] = 1
            doc.cats["Surprise"] = 0
        else:
            doc.cats["Sadness"] = 0
            doc.cats["Joy"] = 0
            doc.cats["Love"] = 0
            doc.cats["Anger"] = 0
            doc.cats["Fear"] = 0
            doc.cats["Surprise"] = 1
        docs.append(doc)
    return docs

num_texts = 500
train_docs = make_docs(train_data[:num_texts])
print(train_docs[3])
doc_bin = DocBin(docs=train_docs)
doc_bin.to_disk("./train.spacy")

valid_docs = make_docs(valid_data[:num_texts])
print(valid_docs[3])
doc_bin = DocBin(docs=valid_docs)
doc_bin.to_disk("./valid.spacy")