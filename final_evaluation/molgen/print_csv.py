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


def calc_avg(perc, keys):
    return np.nanmean([perc[key] for key in keys])


baseline_scores = scores["langmolecules_MolT5-Small"]

baseline_scores["FCD"] = max([scores[key]["FCD"] for key in scores])
baseline_scores["Combo FCD"] = max([scores[key]["FCD"] for key in scores])

print(baseline_scores)
print(scores)

percs = {i: proc_perc(baseline_scores, scores[i]) for i in scores}


cap_list = [
    "BLEU",
    "exact_match",
    "levenshtein",
    "validity",
    "maccs_sim",
    "rdk_sim",
    "morgan_sim",
    "FCD",
]

combo_list = [
    "Combo BLEU",
    "Combo exact_match",
    "Combo levenshtein",
    "Combo validity",
    "Combo maccs_sim",
    "Combo rdk_sim",
    "Combo morgan_sim",
    "Combo FCD",
]

avg_percs_all = {i: calc_avg(percs[i], cap_list) for i in percs}
avg_percs_combo = {i: calc_avg(percs[i], combo_list) for i in percs}

avg_percs = {i: calc_avg(percs[i], cap_list + combo_list) for i in percs}


model_names = {
    "ndhieunguyen": "Lang2mol-diff",
    "peiqz": "BioT5+_large_240513_f165k_text2mol_lm24_test3k_f_extra_gpu8_bsz32_acc2_eps30_lr3e-4_dp0.1_seed42_1984-8928",
    "langmolecules_MolT5-Base": "MolT5-Base",
    "dimitris": "ALMol~10%Data",
    "langmolecules_Meditron": "Meditron",
    "guiyike": "Nano",
    "erikxiong": "PUF",
    "langmolecules_MolT5-Small": "MolT5-Small",
    "mengmeng": "Mistral",
    "danielshao": "SMol+LPM",
    "hecao_bioagent_epoch5": "bioagent_epoch5",
    "hecao_bioagent": "bioagent",
    "avaliev": "PLAIN",
    "protonunfold": "SciMind",
    "qizhipei": "BioT5+_large_240512_f165k_text2mol_lm24_test3k_train_extra_gpu4_bsz32_acc2_eps30_lr3e-4_dp0.1_seed42_118749",
    "langmolecules_MolT5-Large": "MolT5-Large",
}


for key in sorted(avg_percs.items(), key=lambda x: x[1], reverse=True):
    key = key[0]
    score_list = [np.round(avg_percs[key], 2), np.round(avg_percs_all[key], 2)] + [
        np.round(100 * scores[key][prop], 2) for prop in cap_list
    ]
    score_list[-1] = round(score_list[-1] / 100, 2)
    score_list[4] = round(score_list[4] / 100, 2)

    print(key + "," + model_names[key] + "," + ",".join([str(s) for s in score_list]))

print()
print()

for key in sorted(avg_percs.items(), key=lambda x: x[1], reverse=True):
    key = key[0]
    score_list = [np.round(avg_percs[key], 2), np.round(avg_percs_combo[key], 2)] + [
        np.round(100 * scores[key][prop], 2) for prop in combo_list
    ]
    score_list[-1] = round(score_list[-1] / 100, 2)
    score_list[4] = round(score_list[4] / 100, 2)

    print(key + "," + model_names[key] + "," + ",".join([str(s) for s in score_list]))
