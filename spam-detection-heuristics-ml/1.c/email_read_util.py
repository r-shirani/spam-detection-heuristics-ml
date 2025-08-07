import string
import email
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

punctuations = list(string.punctuation)
stopwords = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.PorterStemmer()

trusted_senders = {'trusted@example.com', 'support@bank.com', 'admin@university.edu'}

def flatten_to_string(parts):
    ret = []
    if type(parts) == str:
        ret.append(parts)
    elif type(parts) == list:
        for part in parts:
            ret += flatten_to_string(part)
    elif hasattr(parts, 'get_content_type') and parts.get_content_type() == 'text/plain':
        ret += parts.get_payload()
    return ret

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

def load(path):
    subject, sender, email_text = extract_email_fields(path)
    if not email_text:
        return []

    tokens = word_tokenize(email_text)
    tokens = [i.strip("".join(punctuations)) for i in tokens if i not in punctuations]
    tokens = [w for w in tokens if w not in stopwords]

    if len(tokens) < 2:
        return []

    # create unigrams 
    unigrams = [stemmer.stem(w) for w in tokens]

    # create bigrams(e.x. : 'freemoney', 'clickhere')
    bigrams = [' '.join(pair) for pair in ngrams(unigrams, 2)]

    result = unigrams + bigrams

    if not subject.strip():
        result.append('__NOSUBJECT__')

    if sender and sender.strip().lower() not in trusted_senders:
        result.append('__UNTRUSTED_SENDER__')

    return result
