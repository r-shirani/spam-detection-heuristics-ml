import os
import pickle
import email_read_util as email_read_util
from IPython.display import HTML, display
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')

DATA_DIR = 'D:/SECURITY/trec07p/data/'
LABELS_FILE = 'D:/SECURITY/trec07p/full/index'
TRAINING_SET_RATIO = 0.7
labels = {}
spam_words = set()
ham_words = set()
INVALID_SENDERS = ['unknown', 'noreply', 'mailer-daemon']

# Read the labels
with open(LABELS_FILE) as f:
    for line in f:
        line = line.strip()
        label, key = line.split()
        labels[key.split('/')[-1]] = 1 if label.lower() == 'ham' else 0

# Split corpus into train and test sets
filelist = os.listdir(DATA_DIR)
X_train = filelist[:int(len(filelist)*TRAINING_SET_RATIO)]
X_test = filelist[int(len(filelist)*TRAINING_SET_RATIO):]

# Thresholds
HAM_THRESHOLD = 0.25

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

# --- Evaluation ---
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

        subject, sender = email_read_util.extract_subject_sender(path)
        is_suspected_spam = False

        # Rule 1: Contains blacklisted words
        if set(stems) & blacklist:
            is_suspected_spam = True

        # Rule 2: No subject
        if not subject.strip():
            is_suspected_spam = True

        # Rule 3: Invalid sender
        if any(invalid in sender.lower() for invalid in INVALID_SENDERS):
            is_suspected_spam = True

        # Confusion matrix logic
        if is_suspected_spam:
            if label == 1:
                fp += 1
            else:
                tp += 1
        else:
            if label == 1:
                tn += 1
            else:
                fn += 1

# --- Results ---
conf_matrix = [[tn, fp], [fn, tp]]
print(conf_matrix)

count = tn + tp + fn + fp
percent_matrix = [
    ["{:.1%}".format(tn/count), "{:.1%}".format(fp/count)],
    ["{:.1%}".format(fn/count), "{:.1%}".format(tp/count)]
]
print(percent_matrix)
print("Classification accuracy: {}".format("{:.1%}".format((tp + tn) / count)))