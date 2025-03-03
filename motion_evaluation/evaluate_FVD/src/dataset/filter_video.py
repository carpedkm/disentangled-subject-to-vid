import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')
import pandas as pd
from tqdm import tqdm 
import re

category_file = "k400_category.txt"

# read the file with two columns, the first column is the category id, the second column is the category name
k400_cat_word = []
with open(category_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            category_id, category_name = line.split('\t')
            category_name = category_name.strip()
            
            tokens = word_tokenize(category_name)
            # tag each token with its part of speech
            tagged = nltk.pos_tag(tokens)
            # lemmatize and filter out all the verbs and nouns
            lemmatizer = WordNetLemmatizer()
            cato_words = []
            for word, tag in tagged:
                if tag.startswith('NN'):
                    cato_words.append(lemmatizer.lemmatize(word, pos='n'))
                elif tag.startswith('VB'):
                    cato_words.append(lemmatizer.lemmatize(word, pos='v'))
            # print(cato_words)
            k400_cat_word.append(cato_words)
# print(k400_cat_word)


vid_metadata = "D:\\data\\webvid\\results_10M_train_cleaned.csv"
metadata = pd.read_csv(vid_metadata)
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
filtered_videoid = []
filtered_caption = []
filtered_page_idx = []
filtered_page_dir = []
filtered_duration = []
filtered_contentUrl = []
# for each video metadata, find the words that are in the category list 
for idx, one_v in tqdm(enumerate(metadata['caption'])):
    # print(one_v)
    # print(idx)
    # print(metadata['videoid'][idx])
    # remove all non-alphanumeric characters
    one_v = one_v.replace("\"","")
    sentence = replace_abbreviations(one_v)
    # tokenize the sentence
    tokens = word_tokenize(sentence)
    # tag each token with its part of speech
    tagged = nltk.pos_tag(tokens)
    # lemmatize and filter out all the verbs and nouns
    lemmatizer = WordNetLemmatizer()
    words = []
    for word, tag in tagged:
        if tag.startswith('NN'):
            words.append(lemmatizer.lemmatize(word, pos='n'))
        elif tag.startswith('VB'):
            words.append(lemmatizer.lemmatize(word, pos='v'))
    # check whether each category in k400_cat_word is in words
    for cat in k400_cat_word:
        # if the size of cat is 1, check whether cat is the subset of words 
        if len(cat) == 1:
            if cat[0] in words:
                filtered_videoid.append(metadata['videoid'][idx])
                filtered_caption.append(one_v)
                # filtered_page_idx.append(metadata['page_idx'][idx])
                filtered_page_dir.append(metadata['page_dir'][idx])
                filtered_duration.append(metadata['duration'][idx])
                filtered_contentUrl.append(metadata['contentUrl'][idx])
                # jump out the loop
                break
        # if the size of cat is 2, check whether cat is the subset of words
        else:
            # check at least two words are overlapped
            if len(set(cat).intersection(set(words))) >= 2:
                filtered_videoid.append(metadata['videoid'][idx])
                filtered_caption.append(one_v)
                # filtered_page_idx.append(metadata['page_idx'][idx])
                filtered_page_dir.append(metadata['page_dir'][idx])
                filtered_duration.append(metadata['duration'][idx])
                filtered_contentUrl.append(metadata['contentUrl'][idx])
                break

# write all the filtered results into a csv file
filter_file_output = "D:\\data\\webvid\\results_10M_train_cleaned_k400_filtered.csv"
with open(filter_file_output, 'w', errors="ignore") as f:
    f.write('videoid,name,page_dir,duration,contentUrl\n')
    for idx, one_v in enumerate(filtered_videoid):
        f.write(str(one_v) + ',\"' + str(filtered_caption[idx]).strip().replace("\n","") + '\",' + str(filtered_page_dir[idx]).strip().replace("\n","") + ',' + str(filtered_duration[idx]) + ',' + str(filtered_contentUrl[idx]).strip().replace("\n","") + '\n')

print(len(filtered_videoid))



