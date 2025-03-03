import json
import pandas as pd

# list_file = "/home/qid/blob/getty_video/video/getty_video_list.txt"
# meta_file = "/home/qid/workspace/T2VG/getty_video_1M_meta.csv"
# csv_handler = open(meta_file, "w")
# csv_handler.write("videoid,title,description,tags"+"\n")

# with open(list_file,'r') as f:
#     lines=f.readlines()
#     for line in lines:
#         line = line.rstrip()
#         if line.endswith('json'):
#             path = "/home/qid/blob/getty_video/video/videos/"+line
#             with open(path, "r") as json_f:
#                 des = json.load(json_f)
#                 video_path = line[:-4]+"mp4"
#                 if des['title'] is not None:
#                     tmp_title = des['title'].replace("\"","")
#                 else:
#                     tmp_title = ""
#                 if des['description'] is not None:
#                     tmp_des = des['description'].replace("\"","")
#                 else:
#                     tmp_des = ""
#                 csv_handler.write(video_path+",\"" + tmp_title + "\",\"" + tmp_des + "\",\"")
#                 if des['tags'] is not None:
#                     for i in range(len(des['tags'])-1):
#                         if len(des['tags'][i]) != 0:
#                             csv_handler.write(des['tags'][i].replace("\"","") + ", ")
#                     csv_handler.write(des['tags'][len(des['tags'])-1].replace("\"",""))
#                 csv_handler.write("\"\n")
# csv_handler.close()


# print(len(metadata['description']))
# print(len(metadata['videoid']))



# calculate the number of blank in a string 
def calblank(string_txt):
    blank_num = 0
    for i in range(len(string_txt)):
        if string_txt[i] == " ":
            blank_num += 1
    return blank_num
    

# calculate the average word length in the variables
meta_file = "/home/qid/blob/getty_video/video/results_1M_train.csv"
metadata = pd.read_csv(meta_file)
for i in range(len(metadata['description'])):
    # the number of blank in metadata['description'][i]
    