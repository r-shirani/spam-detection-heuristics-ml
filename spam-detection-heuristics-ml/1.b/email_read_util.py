import string
import email
import nltk
from nltk.tokenize import word_tokenize

punctuations = list(string.punctuation)
stopwords = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.PorterStemmer()

# Combine the different parts of the email into a flat list of strings
def flatten_to_string(parts):
    ret = []
    if type(parts) == str:
        ret.append(parts)
    elif type(parts) == list:
        for part in parts:
            ret += flatten_to_string(part)
    elif parts.get_content_type == 'text/plain':
        ret += parts.get_payload()
    return ret

# Extract subject, sender and body text from a single email file
def extract_email_fields(path):
    with open(path, errors='ignore') as f:
        msg = email.message_from_file(f)
    if not msg:
        return "", "", ""

    subject = msg['Subject'] if msg['Subject'] else ""
    sender = msg['From'] if msg['From'] else ""
    body = ' '.join(m for m in flatten_to_string(msg.get_payload()) if type(m) == str)
    if not body:
        body = ""

    return subject, sender, subject + ' ' + body

# Shortcut for just subject and sender
def extract_subject_sender(path):
    subject, sender, _ = extract_email_fields(path)
    return subject, sender

# Load and tokenize + stem email content
def load(path):
    _, _, email_text = extract_email_fields(path)
    if not email_text:
        return []

    tokens = word_tokenize(email_text)
    tokens = [i.strip("".join(punctuations)) for i in tokens if i not in punctuations]
    if len(tokens) > 2:
        return [stemmer.stem(w) for w in tokens if w not in stopwords]
    return []
