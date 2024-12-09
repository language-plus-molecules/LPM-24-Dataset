import json

import os
from os import listdir
from os.path import isfile, join

import numpy as np

score_dir = "scores/"
onlyfiles = [f for f in listdir(score_dir) if isfile(join(score_dir, f))]

scores = {}

for f in onlyfiles:
    print(f)

    with open(os.path.join(score_dir, f), "r") as score_file:
        s = json.load(score_file)
    scores[f.split(".")[0]] = s

score_dir = "scores_combo/"
onlyfiles = [f for f in listdir(score_dir) if isfile(join(score_dir, f))]

for f in onlyfiles:
    print(f)

    with open(os.path.join(score_dir, f), "r") as score_file:
        s = json.load(score_file)
    for key in s:
        scores[f.split(".")[0]][key] = s[key]


ignore_list = ["exact_match", "Combo exact_match"]
flip_list = ["levenshtein", "FCD", "Combo levenshtein", "Combo FCD"]


def proc_perc(base, inp):
    res = {}
    for key in base:
        if base[key] == 0.0:
            res[key] = 0.0
            continue
        res[key] = (inp[key] - base[key]) * 100
        if key in flip_list:
            res[key] = -res[key] / 100

    return res


def calc_avg(
    perc,
):
    for ig in ignore_list:
        perc.pop(ig)

    vals = [s if not np.isnan(s) else 0.0 for s in list(perc.values())]
    return np.mean(vals)


baseline_scores = scores["langmolecules_MolT5-Small"]

baseline_scores["FCD"] = max([scores[key]["FCD"] for key in scores])
baseline_scores["Combo FCD"] = max([scores[key]["FCD"] for key in scores])

print(baseline_scores)
print(scores)

percs = {i: proc_perc(baseline_scores, scores[i]) for i in scores}

print()
print(percs)

print()


avg_percs = {i: calc_avg(percs[i]) for i in percs}

print(avg_percs)
