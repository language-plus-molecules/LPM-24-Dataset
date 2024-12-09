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


def calc_avg(perc, keys):
    return np.nanmean([perc[key] for key in keys])


baseline_scores = scores["langmolecules_MolT5-Small"]


percs = {i: proc_perc(baseline_scores, scores[i]) for i in scores}


cap_list = ["BLEU-2", "BLEU-4", "Rouge-1", "Rouge-2", "Rouge-L", "Meteor"]

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
    "Combo F-1",
]


avg_percs_cap = {i: calc_avg(percs[i], cap_list) for i in percs}
avg_percs_prop = {i: calc_avg(percs[i], key_list) for i in percs}

avg_percs = {i: calc_avg(percs[i], cap_list + key_list) for i in percs}


model_names = {
    "avaliev": "RAG_SIM_098",
    "qizhipei": "BioT5+_large_ensemble_r",
    "protonunfold": "SciMind",
    "shinnosuke_Ensembled": "Ensembled",
    "shinnosuke_Rank_model_1": "Rank_model_1",
    "hecao": "bioagent",
    "xwk89_mistral_4b9_e1": "mistral_4b9_e1",
    "xwk89_mistral_e1": "mistral_e1",
    "mengmeng": "Mistral",
    "langmolecules_Meditron": "Meditron",
    "dimitris": "ALMol~10%DataTrained",
    "xygui": "MDEG",
    "danielshao": "SMol+LPM",
    "duongttr": "Mol2Lang-VLM",
    "langmolecules_MolT5-Large": "MolT5-Large",
    "bluesky333": "phi3-knowchem-sft-beam1",
    "langmolecules_MolT5-Base": "MolT5-Base",
    "langmolecules_MolT5-Small": "MolT5-Small",
    "guiyike": "yike",
    "peiqz": "BioT5+_large_voting_eval_lm24_mol2text_test_dist_extra_f_extra_tel_6",
}


for key in sorted(avg_percs.items(), key=lambda x: x[1], reverse=True):
    key = key[0]
    score_list = [np.round(avg_percs[key], 2), np.round(avg_percs_cap[key], 2)] + [
        np.round(100 * scores[key][prop], 2) for prop in cap_list
    ]

    print(key + "," + model_names[key] + ",", ",".join([str(s) for s in score_list]))

print()
print()

for key in sorted(avg_percs.items(), key=lambda x: x[1], reverse=True):
    key = key[0]
    score_list = [np.round(avg_percs[key], 2), np.round(avg_percs_prop[key], 2)] + [
        np.round(100 * scores[key][prop], 2) for prop in key_list
    ]

    print(key + "," + model_names[key] + ",", ",".join([str(s) for s in score_list]))
