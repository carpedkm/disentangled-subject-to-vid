# -*- coding: utf-8 -*-
#   Auther: William Zhao    #
# Stay foolish, stay hungry.#
# ------------------------- #

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def analyse_label(labels):
    # load data
    all_des = None
    all_dur = None
    for label in labels:
        label_data = pd.read_csv(label)
        label_des = label_data["name"]
        label_dur = label_data["duration"]
        if all_des is None:
            all_des = label_des
        else:
            all_des = pd.concat([all_des, label_des], ignore_index=True)

        if all_dur is None:
            all_dur = label_dur
        else:
            all_dur = pd.concat([all_dur, label_dur], ignore_index=True)
    # get duration
    all_des_l = all_des.str.split(" ").str.len()
    # get word length
    all_dur = all_dur.str[8:10].astype(int)
    # get word / second
    all_word_p_dur = all_des_l / all_dur
    all_word_p_dur.replace([np.inf, -np.inf], np.nan, inplace=True)
    all_word_p_dur.dropna(how="all", inplace=True)

    all_word_p_dur_lt10 = all_word_p_dur[all_word_p_dur < 10]
    all_des_l_lt50 = all_des_l[all_des_l < 50]

    # hist
    fig, ax = plt.subplots()
    all_des_l.hist(bins=50)
    fig.suptitle("description length (word) hist")
    fig.savefig("des_hist.png")

    fig, ax = plt.subplots()
    all_des_l_lt50.hist(bins=50)
    fig.suptitle("description length (word) lt50 hist")
    fig.savefig("des_hist_lt50.png")

    fig, ax = plt.subplots()
    all_dur.hist(bins=50)
    fig.suptitle("duration (second) hist")
    fig.savefig("dur_hist.png")

    fig, ax = plt.subplots()
    all_word_p_dur.hist(bins=100)
    fig.suptitle("words / second hist")
    fig.savefig("words_per_second_hist.png")

    fig, ax = plt.subplots()
    all_word_p_dur_lt10.hist(bins=100)
    fig.suptitle("words / second lt10 hist")
    fig.savefig("words_per_second_hist_lt10.png")


def main():
    labels = [
        "results_10M_val_cleaned.csv",
        "results_2M_val_cleaned.csv",
        "results_10M_train_cleaned.csv",
        "results_2M_train_cleaned.csv",
    ]
    analyse_label(labels)


if __name__ == "__main__":
    main()
