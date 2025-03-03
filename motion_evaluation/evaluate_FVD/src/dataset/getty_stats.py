import json
import pandas as pd

# calculate the number of blank in a string
def calblank(string_txt):
    # print(string_txt)
    string_txt = str(string_txt)
    blank_num = 0
    for i in range(len(string_txt)):
        if string_txt[i] == " ":
            blank_num += 1
    return blank_num

# calculate the average word length in the variables
meta_file = "/home/qid/blob/getty_video/video/results_1M_train.csv"
metadata = pd.read_csv(meta_file)
sum_blank = 0
for i in range(len(metadata['description'])):
    # sum number of blank in each description
    sum_blank += calblank(metadata['description'][i])
# calculate the average word length in the variables
avg_word_len = sum_blank / len(metadata['description'])
print("average word length is: ", avg_word_len)


sum_comma = 0
# calculate the average comma number in the variables
for i in range(len(metadata['tags'])):  
    sum_comma += metadata['tags'][i].count(',')
# calculate the average comma number in the variables
avg_comma_num = sum_comma / len(metadata['tags'])
print("average comma number is: ", avg_comma_num)

# calculate the average number of char 'a' in the variables
sum_a = 0
for i in range(len(metadata['description'])):
    sum_a += metadata['description'][i].count('a')
# calculate the average number of char 'a' in the variables
avg_a_num = sum_a / len(metadata['description'])
print("average number of char 'a' is: ", avg_a_num)