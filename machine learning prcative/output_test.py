import spacy

nlp = spacy.load("output/model-best")
doc = nlp("I tried so hard for my exams and yet i failed I feel like i am shit and i cannot do anything probably i will not get admission in a good college I feel like trash ")
print(doc.cats)
