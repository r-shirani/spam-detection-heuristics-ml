import os
import pickle
import email_read_util as email_read_util
from IPython.display import HTML, display
import nltk

DATA_DIR = 'D:/SECURITY/trec07p/data/'
LABELS_FILE = 'D:/SECURITY/trec07p/full/index'
TRAINING_SET_RATIO = 0.7

labels = {}
spam_words = set()
ham_words = set()

with open(LABELS_FILE) as f:
    for line in f:
        line = line.strip()
        label, key = line.split()
        labels[key.split('/')[-1]] = 1 if label.lower() == 'ham' else 0

filelist = os.listdir(DATA_DIR)
X_train = filelist[:int(len(filelist)*TRAINING_SET_RATIO)]
X_test = filelist[int(len(filelist)*TRAINING_SET_RATIO):]

# threshhold
HAM_THRESHOLD = 0.50

if not os.path.exists('blacklist.pkl'):
    ham_email_count = sum(1 for filename in X_train if labels.get(filename, None) == 1)

    for filename in X_train:
        path = os.path.join(DATA_DIR, filename)
        if filename in labels:
            label = labels[filename]
            stems = email_read_util.load(path)
            if not stems:
                continue
            if label == 1:
                ham_words.update(stems)
            elif label == 0:
                spam_words.update(stems)

    ham_word_freq = {}
    for filename in X_train:
        if labels.get(filename, None) == 1:
            path = os.path.join(DATA_DIR, filename)
            stems = email_read_util.load(path)
            if stems:
                for word in set(stems): 
                    if word not in ham_word_freq:
                        ham_word_freq[word] = 0
                    ham_word_freq[word] += 1

    ham_threshold_count = int(HAM_THRESHOLD * ham_email_count)

    blacklist = set()

    for word in spam_words:
        if ham_word_freq.get(word, 0) >= ham_threshold_count or word not in ham_words:
            blacklist.add(word)

    pickle.dump(blacklist, open('blacklist.pkl', 'wb'))
else:
    blacklist = pickle.load(open('blacklist.pkl', 'rb'))

print(f'Blacklist of {len(blacklist)} tokens successfully built/loaded')

fp = 0
tp = 0
fn = 0
tn = 0

for filename in X_test:
    path = os.path.join(DATA_DIR, filename)
    if filename in labels:
        label = labels[filename]
        stems = email_read_util.load(path)
        if not stems:
            continue
        stems_set = set(stems)
        if stems_set & blacklist:
            if label == 1:
                fp += 1  # False Positive
            else:
                tp += 1  # True Positive
        else:
            if label == 1:
                tn += 1  # True Negative
            else:
                fn += 1  # False Negative

conf_matrix = [[tn, fp], [fn, tp]]
print("Confusion Matrix:")
print(conf_matrix)

count = tn + tp + fn + fp
percent_matrix = [
    ["{:.1%}".format(tn/count), "{:.1%}".format(fp/count)],
    ["{:.1%}".format(fn/count), "{:.1%}".format(tp/count)]
]
print("Percent Matrix:")
print(percent_matrix)

print("Classification accuracy: {}".format("{:.1%}".format((tp + tn) / count)))