import os
import email_read_util
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

DATA_DIR = 'D:/SECURITY/trec07p/data/'
LABELS_FILE = 'D:/SECURITY/trec07p/full/index'
TRAINING_SET_RATIO = 0.7

# Load labels
labels = {}
with open(LABELS_FILE) as f:
    for line in f:
        line = line.strip()
        label, key = line.split()
        labels[key.split('/')[-1]] = 1 if label.lower() == 'ham' else 0

# Load emails and convert to plain text
def read_email_files():
    X = []
    y = []
    for i in range(len(labels)):
        filename = 'inmail.' + str(i + 1)
        tokens = email_read_util.load(os.path.join(DATA_DIR, filename))
        email_str = ' '.join(tokens)  # convert token list to string
        X.append(email_str)
        y.append(labels[filename])
    return X, y

X, y = read_email_files()

# Split dataset
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, range(len(y)), train_size=TRAINING_SET_RATIO, random_state=2
)

# Vectorize emails
vectorizer = CountVectorizer()
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# Train SVM
classifier = LinearSVC(dual=False, max_iter=20000, tol=1e-4)
classifier.fit(X_train_vector, y_train)
y_pred = classifier.predict(X_test_vector)

# Print evaluation
print(classification_report(y_test, y_pred, target_names=['Spam', 'Ham']))
print('Classification accuracy {:.1%}'.format(accuracy_score(y_test, y_pred)))