import json


file = "scores_props_old/avaliev.json"


with open(file, "r") as score_file:
    s = json.load(score_file)


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

for key in key_list:
    print(key, s["F-1"][key])

print(s["Combo F-1"])
