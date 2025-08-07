import os
import email_read_util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold

DATA_DIR = 'D:/SECURITY/trec07p/data/'
LABELS_FILE = 'D:/SECURITY/trec07p/full/index'
N_SPLITS = 5  # Set number of folds for cross-validation

# Read labels from file
labels = {}
with open(LABELS_FILE) as f:
    for line in f:
        line = line.strip()
        label, key = line.split()
        labels[key.split('/')[-1]] = 1 if label.lower() == 'ham' else 0

# Read and preprocess emails
def read_email_files():
    X = []
    y = []
    for i in range(len(labels)):
        filename = 'inmail.' + str(i + 1)
        tokens = email_read_util.load(os.path.join(DATA_DIR, filename))
        email_str = ' '.join(tokens)  # Join tokens into a single string
        X.append(email_str)
        y.append(labels[filename])
    return X, y

# Prepare data
X, y = read_email_files()

# Vectorize the text
vectorizer = CountVectorizer()
X_vector = vectorizer.fit_transform(X)

# Set up the classifier and cross-validation
classifier = MultinomialNB()
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=2)
y_pred = cross_val_predict(classifier, X_vector, y, cv=cv)

# Output evaluation results
print(classification_report(y, y_pred, target_names=['Spam', 'Ham']))
print('Classification accuracy: {:.1%}'.format(accuracy_score(y, y_pred)))

# Calculate FPR
cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn)
print('False Positive Rate (FPR): {:.2%}'.format(fpr))