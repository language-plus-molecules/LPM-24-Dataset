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


score_dir = "scores_props/"
onlyfiles = [f for f in listdir(score_dir) if isfile(join(score_dir, f))]


key_list = [
    "Overall",
    "Biomedical",
    "Human Interaction and Organoleptics",
    "Agriculture and Industry",
    "Light and electricity",
    "Agriculture and Industry__icides",
    "Human Interaction and Organoleptics__toxin",
    "Light and electricity__lights",
    "Light and electricity__electros",
    "Biomedical__inhibitors",
    "Biomedical__antis",
    "Biomedical__modulators",
    "Biomedical__antagonists",
    "Biomedical__treatments",
    "Biomedical__agonists",
    "Biomedical__cancer",
    "Biomedical__disease",
]


for f in onlyfiles:
    if f == "test.json":
        continue
    print(f)

    with open(os.path.join(score_dir, f), "r") as score_file:
        s = json.load(score_file)
    for key in key_list:
        scores[f.split(".")[0]][key] = s["F-1"][key]
    scores[f.split(".")[0]]["Combo F-1"] = s["Combo F-1"]


ignore_list = []
flip_list = []


def proc_perc(base, inp):
    res = {}
    for key in base:

        res[key] = (inp[key] - base[key]) * 100
        if key in flip_list:
            res[key] = -res[key]

    return res


def calc_avg(perc):
    for ig in ignore_list:
        perc.pop(ig)

    return np.nanmean(list(perc.values()))


baseline_scores = scores["langmolecules_MolT5-Small"]

print(baseline_scores)
print(scores)

percs = {i: proc_perc(baseline_scores, scores[i]) for i in scores}

print()
print(percs)

print()


avg_percs = {i: calc_avg(percs[i]) for i in percs}

print(avg_percs)
