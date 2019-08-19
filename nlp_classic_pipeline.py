import re
import string

from collections import Counter
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
# from spacy.lang.fr.stop_words import STOP_WORDS
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk import stem
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

import matplotlib.pyplot as plt
from adjustText import adjust_text

# %matplotlib inline

pca = PCA(n_components=2)
nlp = spacy.load('en_core_web_lg')
shortword = re.compile(r'\W*\b\w{1,3}\b')


def preprocess_column(self, column_name, to_delete=None, digit=True, short_words=True):
    '''
        *Clean a corpus in a DataFrame*

        :param column_name: Column to clean
        :param to_delete: Text to delete
        :param digit: Remove digits
        :param short_words: Remove words having length between 1 and 3
        :type column_name: str
        :type to_delete: tuple
        :type digit: bool
        :type short_words: bool
        :rtype: list
    '''

    print('Preprocessing the column')
    self = self.dropna(subset=[column_name])
    self = self.drop_duplicates([column_name])
    print('Cleaning')
    self[column_name] = self[column_name].apply(lambda x: cleaning(x, to_delete, digit, short_words))
    print('Preprocessing over')
    return self[column_name].tolist()


pd.DataFrame.preprocess_column = preprocess_column


def cleaning(text, to_delete=None, digit=True, short_words=True):
    '''
        *Clean a sentence*

        :param text: Text to clean
        :param to_delete: Text to delete
        :param digit: Remove digits
        :param short_words: Remove words having length between 1 and 3
        :type text: str
        :type to_delete: tuple
        :type digit: bool
        :type short_words: bool
        :rtype: str
    '''

    regex = re.compile('[%s]' % re.escape(string.punctuation))
    text = text.lower()
    if digit:
        text = re.sub('\d+', '', text)

    if to_delete:
        for words in zip(*to_delete):
            text = text.replace(*words)

    text = ' '.join([word for word in text.split(' ') if word.lower() not in STOP_WORDS])
    if short_words:
        text = shortword.sub('', text)
    text = regex.sub(' ', text)
    text = text.strip()
    return text


def create_stemmed_corpus(corpus):
    '''
        *Perform stemming on a corpus*

        :param corpus: Corpus of text
        :type corpus: list
        :rtype: list
    '''
    print('Doing some stemming')
    corpus = [word_tokenize(string) for string in corpus]
    corpus = [[ps.stem(w) for w in string] for string in corpus]
    corpus = [' '.join(string) for string in corpus]
    return corpus


def create_lemmatized_corpus(corpus):
    '''
        *Perform lemmatization on a corpus*

        :param corpus: Corpus of text
        :type corpus: list
        :rtype: list
    '''

    print('Doing some lemmatization')
    corpus = [word_tokenize(string) for string in corpus]
    corpus = [[wordnet_lemmatizer.lemmatize(w, pos='v') for w in string] for string in corpus]
    corpus = [[wordnet_lemmatizer.lemmatize(w, pos='n') for w in string] for string in corpus]
    corpus = [' '.join(string) for string in corpus]
    return corpus


def create_stemmed_corpus_fra(corpus):
    '''
        *Perform stemming on a French corpus*

        :param corpus: Corpus of text
        :type corpus: list
        :rtype: list
    '''

    print('Doing some stemming')
    stemmer = stem.Regexp('s$|es$|era$|erez$|ions$| <etc> ')
    corpus = [word_tokenize(string) for string in corpus]
    corpus = [[ps.stem(w) for w in string] for string in corpus]
    corpus = [' '.join(string) for string in corpus]
    return corpus


def create_lemmatized_corpus_fra(corpus):
    '''
        *Perform lemmatization on a French corpus*

        :param corpus: Corpus of text
        :type corpus: list
        :rtype: list
    '''

    print('Doing some lemmatization')
    corpus = [nlp(string) for string in corpus]
    print('nlp over')
    corpus = [[token.lemma_ for token in string] for string in corpus]
    print('lemma over')
    corpus = [' '.join(string) for string in corpus]
    return corpus


def delete_generator(to_delete, to_replace=''):
    '''
        *Create the tuple of text to remove*

        :param to_delete: Text to remove
        :param to_replace: Text to replace
        :type to_remove: tuple
        :type to_replace: str
        :rtype: tuple
    '''

    to_replace = (to_replace,) * len(to_delete)
    to_delete = (to_delete, to_replace)
    return to_delete


def tfidf_and_count(corpus):
    '''
        *Perform Tfidf and count the features*

        :param corpus: Text to analyze
        :type corpus: list
        :rtype: tuple
    '''

    vectorizer = TfidfVectorizer(max_df=0.95)
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names()
    counts = X.sum(axis=0).A1
    freq_distribution = Counter(dict(zip(feature_names, counts)))
    return freq_distribution


def vectorize_and_plot(freq_distribution, max_words=40):
    '''
        *Perform word vectorization, pca and plot*

        :param freq_distribution: Distribution of word frequence from Tfidf
        :param max_words: Number max of words to analyze
        :type freq_distribution: counter
        :type max_words: int
    '''

    # Get the 40 most common words and setup areas fot the plot
    words, areas = list(zip(*freq_distribution.most_common(max_words)))
    areas = list((item ** 2) * 50 for item in areas)
    print(words)

    # Get vector of the top words and apply PCA to have them in 2D
    words_vector = [nlp(word).vector for word in words]
    pca.fit(words_vector)
    word_vecs_2d = pca.transform(words_vector)
    colors = [int(i[0] % 23) for i in word_vecs_2d]

    # Plot
    plt.figure(figsize=(11, 8))
    plt.scatter(word_vecs_2d[:, 0], word_vecs_2d[:, 1], s=areas, c=colors, cmap='hsv', edgecolors='black', alpha=0.3)
    texts = []
    for word, coord in zip(words, word_vecs_2d):
        x, y = coord
        t = plt.text(x, y, word, size=13)
        texts.append(t)

    adjust_text(texts, only_move={'points': 'y', 'texts': 'y'}, arrowprops=dict(arrowstyle="-", color='b', lw=0))
    plt.savefig('topic.png', bbox_inches='tight')
    plt.show()

    bar = pd.DataFrame({"words": words, "frequency": areas})
    plt.figure(figsize=(10, 7))
    #import seaborn as sns
    #ax = sns.barplot(x='words', y='frequency', data=bar, palette="Blues_d")
    #plt.xticks(rotation=90)
    #plt.ylabel("Frequency", fontsize=17)
    #plt.tick_params(labelsize=15)
    #plt.savefig('topic_frequency_V2.png', bbox_inches='tight')

if __name__ == '__main__':
    # Read raw text and preprocess it to create a corpus
    df = pd.read_csv('D:/Users/erwan.lecovec/Downloads/WPy-3670/notebooks/student.csv', sep=';')
    df = df.astype('unicode')
    column_name = 'liste_reponsability'
    # to_delete = delete_generator(('pppppp', 'llllll', 'ffffff', 'soap', 'quote', 'noncontributori', 'noncontributari', 'noncontribitori', 'noncontibutori', 'noincontributori', 'nnoncontributori','uncontributori'))
    corpus = df.preprocess_column(column_name)
    corpus = create_lemmatized_corpus(corpus)
    freq_distribution = tfidf_and_count(corpus)
    vectorize_and_plot(freq_distribution)
