from sklearn.feature_extraction.text import TfidfVectorizer

from functions import load_model, check_contents_spam_or_ham

loaded_model = load_model('mnb')
#print(loaded_model)

text1 = ["I'm good, how are you?"]
text2 = ["No. I meant the calculation is the same. That I'll call later"]
text3 = ["Had your contract mobile 11 Mnths? Latest Motorola Now"]
text4 = ["WINNER!! You just won a free ticket to Paris. Send your details, on paris@gmail.com"]
text5 = ["Hello, you are winner of special price. BONUS. Contact as on www.google.com"]

vectorizer = TfidfVectorizer()
data = [text1, text2, text3, text4, text5]
vectorizer.fit_transform(data)

check_contents_spam_or_ham(vectorizer, loaded_model, text1)
check_contents_spam_or_ham(vectorizer, loaded_model, text2)
check_contents_spam_or_ham(vectorizer, loaded_model, text3)
check_contents_spam_or_ham(vectorizer, loaded_model, text4)
check_contents_spam_or_ham(vectorizer, loaded_model, text5)
