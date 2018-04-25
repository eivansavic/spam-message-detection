import warnings

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud

from api_twitter import get_tweets
from functions import remove_punctuation_and_stop_words, find_words_by_type, \
    init_classifiers, train, predict, load_model, check_contents_spam_or_ham, check, stemming, display_word_cloud, \
    save_model, display_prediction, lemmating

warnings.filterwarnings('ignore')

data = pd.read_csv('data/spam.csv', encoding='latin-1')
data.head()

data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v2": "text", "v1": "label"})

# print(data[1990:2000])

data['label'].value_counts()
data['length'] = data['text'].map(lambda x: len(x))

data.hist(column='length', bins=50, figsize=(10, 7))
data.hist(column='length', by='label', bins=100, figsize=(20, 7))

# uzimamo sve reci iz svih ham/spam recenica
ham_words = find_words_by_type(data, 'ham')
spam_words = find_words_by_type(data, 'spam')

# prikaz reci u konzoli
# print('\nHam words: ' + ham_words)
# print('\nSpam words: ' + spam_words)

# WordCloud generise sliku na osnovu ham/spam reci
spam_word_cloud = WordCloud(width=500, height=300).generate(spam_words)
ham_word_cloud = WordCloud(width=500, height=300).generate(ham_words)

# prikaz slike
display_word_cloud(spam_word_cloud)
display_word_cloud(ham_word_cloud)

data = data.replace(['ham', 'spam'], [0, 1])

# ispis prvih 10 redova od data
# print(data.head(10))

# izbacivanje znaka interpukcije i stop reci (is, not, are, has...)
data['text'] = data['text'].apply(remove_punctuation_and_stop_words)

# primenjivanje stematizacije i lematizacije
data['text'] = data['text'].apply(stemming)
# data['text'] = data['text'].apply(lemmating)

# print(data.head(10))

text = pd.DataFrame(data['text'])
label = pd.DataFrame(data['label'])

# --Convert words to vectors manual--
# all_words_with_count = count_word_appearing_in_data_set(text)
# print("Total words in data set: ", len(all_words_with_count))
# vocabulary = sort(all_words_with_count, 'desc')
# print(vocab[:60])
# vocabulary_size = len(vocabulary)
# word_indexes = mapping_from_words_to_index(vocabulary)
# word_vectors = mapping_from_text_to_vector(word_indexes, text, vocabulary_size)

# --Converting words to vectors using TFIDF Vectorizer--
# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()
word_vectors = vectorizer.fit_transform(data['text'])

# features = vectors
features = word_vectors

x_train, x_test, y_train, y_test = train_test_split(features, data['label'], test_size=0.15, random_state=111)
# print(x_train.shape)
# print(x_test.shape)

classifiers = init_classifiers()

prediction_scores_word_vectors = []
for k, v in classifiers.items():
    train(v, x_train, y_train)
    prediction = predict(v, x_test)
    prediction_scores_word_vectors.append((k, [accuracy_score(y_test, prediction)]))

# scores = cross_val_score(svm.SVC, features, data['label'], cv=5)
# print(scores)

predictions_word_vectors = pd.DataFrame.from_items(prediction_scores_word_vectors, orient='index', columns=['SCORE'])
print(predictions_word_vectors)

display_prediction(predictions_word_vectors)
model_for_save = classifiers["NB"]  # 0. 'SVC': svc; 1. 'KN': knc; 2. 'NB': mnb, 3. 'DT': dtc, 4. 'LR': lrc, 5. 'RF': rfc
save_model(model_for_save, 'mnb')

loaded_model = load_model('mnb')
# print(loaded_model)

tweets = ["I'm good, how are you?",
          "No. I meant the calculation is the same. That I'll call later",
          "Had your contract mobile 11 Mnths? Latest Motorola Now",
          "WINNER!! You just won a free ticket to Paris. Send your details, on paris@gmail.com",
          "Hello, you are winner of special price. BONUS. Contact as on www.google.com",
          "K I will be sure to get up after noon and see what is what",
          "I am fine, what's about you?",
          "YES. You Win a lot of money! Call us immediately!",
          "Had your contract mobile 11 Mnths? Latest Samsung Now, only today!",
          "Lucky man!! You just won a five ticket for Best Event in New York. Call us on 0626115543",
          "Go on www.facebook.com and add me for friend. Call me!",
          "No, i'm busy today. Maybe tommorow?",
          "That's nice. I have new job for you. Are you in?",
          "Your idea is amazing. We can try it tomorrow.",
          "If you order now, you will get 50% of bonus.",
          "SALE SALE SALE!! You won a billion dollars!",
          "Free credits for games! 50% sales",
          "I think this is fake"]

# spam = 10

# ham = 8

tweets = get_tweets("tim_cook", 4)

dictionary = check_contents_spam_or_ham(vectorizer, loaded_model, tweets)

print(dictionary)
