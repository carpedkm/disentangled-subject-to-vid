import jsonlines
import json
import tqdm


filepath = "D:\\data\\hdvila_meta\\hdvila_meta\\hdvila_clip_text_100m.jsonl"
meta_file = "D:\\data\\hdvila_meta\\hdvila_meta\\hdvila_meta.csv"
csv_handler = open(meta_file, "w")
csv_handler.write("videoid,text"+"\n")

# read the jsonl file, exact the field "clip_id" and "text" into a csv file
with open(filepath,'r', errors="ignore") as f:
    for l in tqdm.tqdm(jsonlines.Reader(f)):
        # print(l)
        name_split = l['clip_id'].split(".")
        vpath = name_split[0]+"/"+l['clip_id']
        csv_handler.write(vpath+",\"" + l['text'].replace("\"","") + "\"\n")

csv_handler.close()
        



