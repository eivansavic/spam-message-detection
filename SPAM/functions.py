import string
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def stemming(text):
    porter_steemer = PorterStemmer()
    text = porter_steemer.stem(text)
    return text


def lemmating(text):
    word_net_lemmatizer = WordNetLemmatizer()
    text = word_net_lemmatizer.lemmatize(text)
    return text


def remove_punctuation_and_stop_words(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)


def find_words_by_type(data, type):
    words = ''
    for val in data[data['label'] == type].text:
        text = val.lower()
        tokens = nltk.word_tokenize(text)
        for token in tokens:
            words += (token + ' ')

    return words


def display_word_cloud(word_cloud):
    plt.figure(figsize=(10, 8), facecolor='w')
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    return


def count_word_appearing_in_data_set(text):
    total_counts = Counter()
    for i in range(len(text)):
        for word in text.values[i][0].split(" "):
            total_counts[word] += 1

    return total_counts


def sort(words, type):
    reverse = False
    if type == 'desc':
        reverse = True

    return sorted(words, key=words.get, reverse=reverse)


def mapping_from_words_to_index(vocabulary):
    word_to_index = {}
    for i, word in enumerate(vocabulary):
        word_to_index[word] = i

    return word_to_index


def text_to_vector(word_to_index, text, vocabulary_size):
    word_vector = np.zeros(vocabulary_size)
    for word in text.split(" "):
        if word_to_index.get(word) is None:
            continue
        else:
            word_vector[word_to_index.get(word)] += 1

    return np.array(word_vector)


def mapping_from_text_to_vector(word_to_index, text, vocabulary_size):
    word_vectors = np.zeros((len(text), vocabulary_size), dtype=np.int_)
    for ii, (_, text_) in enumerate(text.iterrows()):
        word_vectors[ii] = text_to_vector(word_to_index, text_[0], vocabulary_size)

    return word_vectors


def init_classifiers():
    # linear, radial, sigmoid, polynomial, rdf-treba
    svc = SVC(kernel='sigmoid', gamma=0.07, C=1000) # c = 1 default
    knc = KNeighborsClassifier(n_neighbors=49)
    mnb = MultinomialNB(alpha=0.001)
    dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111)
    lrc = LogisticRegression(solver='liblinear', penalty='l1')
    rfc = RandomForestClassifier(n_estimators=31, random_state=111)

    # return {'SVC': svc, 'KN': knc, 'NB': mnb, 'DT': dtc, 'LR': lrc, 'RF': rfc}
    return {'NB': mnb}


def train(clf, features, targets):
    clf.fit(features, targets)


def predict(clf, features):
    return clf.predict(features)


def display_prediction(predictions_word_vectors):
    predictions_word_vectors.plot(kind='bar', ylim=(0.9, 1.0), figsize=(9, 7), align='center', colormap="Accent")
    plt.xticks(pandas.np.arange(6), predictions_word_vectors.index)
    plt.ylabel('Accuracy Score')
    plt.title('Distribution by Classifier - Word Vectors')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    return


def save_model(model, model_name):
    joblib.dump(model, 'model_' + model_name + '.pkl')
    print("Model " + model_name + " saved")


def load_model(model_name):
    model = joblib.load('model_' + model_name + '.pkl')
    print("Model " + model_name + " loaded")
    return model


def get_percent(number, amount):
    return (number / amount) * 100


def check(p):
    if p == 1:
        print("Message is SPAM")
        return 1
    else:
        print("Message is HAM")
        return 0


def check_contents_spam_or_ham(vectorizer, model, contents):
    ham_numbers = 0
    spam_numbers = 0

    integers = vectorizer.transform(contents)
    predicts = model.predict(integers)

    dictionary = {}
    i = 0
    for p in predicts:
        flag = check(p)
        dictionary[contents[i]] = flag
        i += 1

        if flag == 1:
            spam_numbers += 1
        else:
            ham_numbers += 1

    spam_percent = get_percent(spam_numbers, ham_numbers + spam_numbers)
    print('SPAM: ' + str(spam_percent) + '% & HAM: ' + str(100 - spam_percent) + '%')
    return dictionary
