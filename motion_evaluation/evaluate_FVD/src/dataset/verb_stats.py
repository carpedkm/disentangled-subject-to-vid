import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
import collections
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from tqdm import tqdm 


metadata_fp = "D:\\data\\webvid\\results_2M_val_cleaned.csv"
metadata = pd.read_csv(metadata_fp)
metadata['caption'] = metadata['name']
del metadata['name']


# patterns that used to find or/and replace particular chars or words
# to find chars that are not a letter, a blank or a quotation
pat_letter = re.compile(r'[^a-zA-Z \']+')
# to find the 's following the pronouns. re.I is refers to ignore case
pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
# to find the 's following the letters
pat_s = re.compile("(?<=[a-zA-Z])\'s")
# to find the ' following the words ending by s
pat_s2 = re.compile("(?<=s)\'s?")
# to find the abbreviation of not
pat_not = re.compile("(?<=[a-zA-Z])n\'t")
# to find the abbreviation of would
pat_would = re.compile("(?<=[a-zA-Z])\'d")
# to find the abbreviation of will
pat_will = re.compile("(?<=[a-zA-Z])\'ll")
# to find the abbreviation of am
pat_am = re.compile("(?<=[I|i])\'m")
# to find the abbreviation of are
pat_are = re.compile("(?<=[a-zA-Z])\'re")
# to find the abbreviation of have
pat_ve = re.compile("(?<=[a-zA-Z])\'ve")

def replace_abbreviations(text):
    new_text = text
    new_text = pat_letter.sub(' ', text).strip().lower()
    new_text = pat_is.sub(r"\1 is", new_text)
    new_text = pat_s.sub("", new_text)
    new_text = pat_s2.sub("", new_text)
    new_text = pat_not.sub(" not", new_text)
    new_text = pat_would.sub(" would", new_text)
    new_text = pat_will.sub(" will", new_text)
    new_text = pat_am.sub(" am", new_text)
    new_text = pat_are.sub(" are", new_text)
    new_text = pat_ve.sub(" have", new_text)
    new_text = new_text.replace('\'', ' ')
    return new_text

words_box = []
# for each sentence in metadata['caption'], tokenize it to count the number of verbs
for sentence in tqdm(metadata['caption']):
    # remove all non-alphanumeric characters
    sentence = replace_abbreviations(sentence)
    # tokenize the sentence
    tokens = word_tokenize(sentence)
    # tag each token with its part of speech
    tagged = nltk.pos_tag(tokens)
    # lemmatize and filter out all the verbs
    lemmatizer = WordNetLemmatizer()
    verbs = [lemmatizer.lemmatize(word, pos='v') for word, pos in tagged if pos.startswith('VB')]
    words_box.extend(verbs)
dataset_verbs = collections.Counter(words_box)
print(dataset_verbs)
print(len(dataset_verbs))
# get all the verbs in wordnet
from nltk.corpus import wordnet
verbs = list(set([word for word in wordnet.all_lemma_names(pos='v', lang='eng')]))
print(len(verbs))
# get the verbs that are not in wordnet
print(set(words_box) - set(verbs))
print(len(set(words_box) - set(verbs)))

# get all the collocations in the dataset
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(words_box)
# only bigrams that appear 3+ times
finder.apply_freq_filter(3)
# return the 10 n-grams with the highest PMI
print(finder.nbest(bigram_measures.pmi, 10))

# get all the collocations in the wordnet
from nltk.corpus import wordnet
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(verbs)
# only bigrams that appear 3+ times
finder.apply_freq_filter(3)
# return the 10 n-grams with the highest PMI
print(finder.nbest(bigram_measures.pmi, 10))


