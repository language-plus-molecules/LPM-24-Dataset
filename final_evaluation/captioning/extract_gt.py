print("In the scoring function")

import sys
import os

from os import listdir

import pickle
import argparse
import csv
import json

import argparse
import csv

import os.path as osp

import numpy as np

from transformers import BertTokenizerFast

import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from collections import defaultdict

import copy

import io
import zipfile

from os import listdir
from os.path import isfile, join

import random

from tqdm import tqdm

# from rdkit import RDLogger
# RDLogger.DisableLog('rdApp.*')

# print('Imported packages')


score_dir = "prop_files/"


def proc_line(line):
    return line[6:].strip() if line.startswith("[CLS] ") else line.strip()


def flatten(dictionary, separator="__"):
    rv = []
    for key in dictionary:
        for prop in dictionary[key]:
            if isinstance(prop, dict):
                rv += [key + separator + s for s in flatten(prop, separator=separator)]
            elif isinstance(prop, str):
                rv.append(key + separator + prop)
            else:
                zz
    return rv


def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def unflatten(dictionary, separator="__"):
    rv = {}
    for key in dictionary:
        if "__" in key:
            spl = key.split(separator)
            nested_set(rv, spl, dictionary[key])
        else:
            rv[key] = dictionary[key]

    return rv


def flatten_float(dictionary, separator="__"):
    rv = []
    for key in dictionary:
        prop = dictionary[key]
        if isinstance(prop, dict):
            rv += [
                (key + separator + s[0], s[1])
                for s in flatten_float(prop, separator=separator)
            ]
        # elif isinstance(prop, str):
        #    rv.append(key + separator + prop)
        elif isinstance(prop, float):
            rv.append((key, prop))
        elif prop is None:
            continue
        else:
            print(type(prop), prop)
            zz
    return rv


def zero_division(n, d):
    return n / d if d else None


# https://stackoverflow.com/questions/3847386/how-to-test-if-a-list-contains-another-list-as-a-contiguous-subsequence
def contains(small, big):
    for i in range(len(big) - len(small) + 1):
        for j in range(len(small)):
            if big[i + j] != small[j]:
                break
        else:
            return True
    return False


def evaluate(
    gt_caps,
    ot_caps,
    smis,
    text_model="allenai/scibert_scivocab_uncased",
    text_trunc_length=512,
):

    outputs = [
        (smi, proc_line(a), proc_line(b)) for a, b, smi in zip(gt_caps, ot_caps, smis)
    ]

    text_tokenizer = BertTokenizerFast.from_pretrained(text_model)

    def process(outputs, prop_list):

        props = dict()  # defaultdict(list)

        prop_list_tok = [
            (text_tokenizer.tokenize(p.split("__")[-1].lower()), p) for p in prop_list
        ]

        for smi, gt, out in tqdm(outputs):
            props[smi] = []

            gtl = text_tokenizer.tokenize(gt.lower())
            # outl = text_tokenizer.tokenize(out.lower())

            # print(smi)

            for pl, prop in prop_list_tok:

                gtc = contains(pl, gtl)
                # outc = contains(pl, outl)
                # if outc: print(smi, prop)

                if gtc:
                    props[smi].append(prop)

        return props

    nested_props = json.load(open("nested_props.json", encoding="utf-8"))

    flattened_props = flatten(nested_props, separator="__")
    flattened_props = set(flattened_props)

    props = process(outputs, flattened_props)

    # print(props)

    return props


def create_scores(zip_file, out_name):
    # input_dir = sys.argv[1]
    # output_dir = sys.argv[2]

    # reference_dir = os.path.join('/app/input/', 'ref', 'reference_data_molgen')
    # prediction_dir = os.path.join('/app/input/', 'res')

    # print('Checking dirs')
    # print('ref dir:', listdir(reference_dir))
    # print('pred dir:', listdir(prediction_dir))
    # print('out dir:', listdir(score_dir))

    truth_file = "eval-text.txt"
    truth = open(truth_file).readlines()
    # truth_dict = {}
    truth_caps = []
    truth_smis = []
    for line in truth:
        smi, desc, _ = line.split("\t")
        truth_caps.append(desc)
        truth_smis.append(smi)

    # submission_answer_file = os.path.join(prediction_dir, "submit.txt")
    zipf = zipfile.ZipFile(zip_file)
    ot_caps = zipf.open("submit.txt").readlines()

    # ot_smis = open(submission_answer_file).readlines()
    ot_caps = [smi.decode("utf-8").strip() for smi in ot_caps]

    # assert len(truth_caps) + 134 + 3 == len(ot_caps), "Different number of ground truth and predictions." #134 and 3 are FDA and SAB molecules

    print("Predicted and Reference SMILES lists read.")

    print("Calculating string metrics.")

    props = evaluate(truth_caps, ot_caps, truth_smis)

    with open(os.path.join(score_dir, out_name + ".json"), "w") as score_file:
        score_file.write(json.dumps(props))


sub_dir = "submissions/"
onlyfiles = [f for f in listdir(sub_dir) if isfile(join(sub_dir, f))]

random.shuffle(onlyfiles)

for zip_file in onlyfiles:
    # print(zip_file)
    # sys.stdout.flush()
    if zip_file == "test.zip":
        continue

    # zip_file = 'test.zip'

    # if os.path.exists(join(score_dir, zip_file.split('.')[0] + '.json')): continue

    create_scores(sub_dir + zip_file, out_name="ground_truth")
    break
