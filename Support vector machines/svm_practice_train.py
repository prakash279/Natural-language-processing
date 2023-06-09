import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Read the CSV file
data = pd.read_csv('Stess_data.csv')

# Extract text and labels
text = data['text']
labels = data['label']

# Split the data into training and validation sets
text_train, text_val, y_train, y_val = train_test_split(text, labels, test_size=0.75, random_state=42)

# Text preprocessing
vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()

X_train = tfidf_transformer.fit_transform(vectorizer.fit_transform(text_train))
X_val = tfidf_transformer.transform(vectorizer.transform(text_val))

# Initialize the SVM classifier
clf = svm.SVC()

# Train the SVM model
clf.fit(X_train, y_train)

# Predict the labels for validation data
y_pred = clf.predict(X_val)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)


# Preprocess the new sentence
new_sentence = "I can't tell anyone how I am feeling I am exhausted by people's presence sometimes I fell like I want to be with people and sometimes It is shit I talked about this to my teacher I just want to endlessly cry about it and don't know why do I keep thinking about my old pals."
preprocessed_sentence = tfidf_transformer.transform(vectorizer.transform([new_sentence]))

# Predict the label for the new sentence
predicted_label = clf.predict(preprocessed_sentence)

print("Predicted Label:", predicted_label)
