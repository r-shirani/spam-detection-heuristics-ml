import os
import email_read_util
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report , confusion_matrix

DATA_DIR = 'D:/SECURITY/trec07p/data/'
LABELS_FILE = 'D:/SECURITY/trec07p/full/index'

TRAINING_SET_RATIO = 0.7
labels = {}
# Read the labels
with open(LABELS_FILE) as f:
    for line in f:
        line = line.strip()
        label, key = line.split()
        labels[key.split('/')[-1]] = 1 if label.lower() == 'ham' else 0

def read_email_files():
    X = []
    y = [] 
    for i in range(len(labels)):
        filename = 'inmail.' + str(i+1)
        _, _, email_str = email_read_util.extract_email_fields(
            os.path.join(DATA_DIR, filename))
        X.append(email_str)  # Make sure this is a string, not a tuple
        y.append(labels[filename])
    return X, y


X, y = read_email_files()

X_train, X_test, y_train, y_test, idx_train, idx_test = \
    train_test_split(X, y, range(len(y)), 
    train_size=TRAINING_SET_RATIO, random_state=2)


vectorizer = CountVectorizer()
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# Initialize the classifier and make label predictions
mnb = MultinomialNB()
mnb.fit(X_train_vector, y_train)
y_pred = mnb.predict(X_test_vector)

# Print results
print(classification_report(y_test, y_pred, target_names=['Spam', 'Ham']))
print('Classification accuracy {:.1%}'.format(accuracy_score(y_test, y_pred)))
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()  
fpr = fp / (fp + tn)
print('False Positive Rate (FPR): {:.2%}'.format(fpr))