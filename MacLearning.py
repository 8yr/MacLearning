import os
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if not char.isdigit()])
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

def load_data(directory, labels):
    texts = []
    label_list = []
    for label in labels:
        folder = os.path.join(directory, label)
        for filename in os.listdir(folder):
            if filename.endswith('.txt'):
                with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    cleaned_text = clean_text(text)
                    texts.append(cleaned_text)
                    label_list.append(label)
    return texts, label_list

directory = 'Docs'
labels = ['Administrative', 'Financial', 'Specialized']

texts, label_list = load_data(directory, labels)

vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(texts)

encoder = LabelEncoder()
y = encoder.fit_transform(label_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=1)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

model = SVC(kernel='linear')
model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_, zero_division=1, labels=[0, 1, 2]))

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(model, X, y, cv=cv)
print(f'Cross-Validation Accuracy: {cross_val_scores.mean() * 100:.2f}%')

def classify_document(document):
    document_cleaned = clean_text(document)
    document_vector = vectorizer.transform([document_cleaned])
    prediction = model.predict(document_vector)
    return encoder.inverse_transform(prediction)[0]

def read_document_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

file_path = 'test.txt'
new_document = read_document_from_file(file_path)
print(f'classify: {classify_document(new_document)}')