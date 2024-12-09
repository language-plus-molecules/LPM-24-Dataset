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

import io
import zipfile

from os import listdir
from os.path import isfile, join


def proc_line(line):
    return line[6:].strip() if line.startswith("[CLS] ") else line.strip()


def evaluate(
    gt_caps,
    ot_caps,
    text_model="allenai/scibert_scivocab_uncased",
    text_trunc_length=512,
):
    outputs = [(proc_line(a), proc_line(b)) for a, b in zip(gt_caps, ot_caps)]

    text_tokenizer = BertTokenizerFast.from_pretrained(text_model)

    bleu_scores = []
    meteor_scores = []

    references = []
    hypotheses = []

    for i, (gt, out) in enumerate(outputs):

        gt_tokens = text_tokenizer.tokenize(
            gt, truncation=True, max_length=text_trunc_length, padding="max_length"
        )
        gt_tokens = list(filter(("[PAD]").__ne__, gt_tokens))
        gt_tokens = list(filter(("[CLS]").__ne__, gt_tokens))
        gt_tokens = list(filter(("[SEP]").__ne__, gt_tokens))

        out_tokens = text_tokenizer.tokenize(
            out, truncation=True, max_length=text_trunc_length, padding="max_length"
        )
        out_tokens = list(filter(("[PAD]").__ne__, out_tokens))
        out_tokens = list(filter(("[CLS]").__ne__, out_tokens))
        out_tokens = list(filter(("[SEP]").__ne__, out_tokens))

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    _meteor_score = np.mean(meteor_scores)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])

    rouge_scores = []

    references = []
    hypotheses = []

    for i, (gt, out) in enumerate(outputs):

        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    rouge_1 = np.mean([rs["rouge1"].fmeasure for rs in rouge_scores])
    rouge_2 = np.mean([rs["rouge2"].fmeasure for rs in rouge_scores])
    rouge_l = np.mean([rs["rougeL"].fmeasure for rs in rouge_scores])
    return bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score


def create_scores(zip_file, out_name):
    score_dir = "scores/"

    truth_file = "eval-text.txt"
    truth = open(truth_file).readlines()
    truth_caps = []
    for line in truth:
        smi, desc, _ = line.split("\t")
        truth_caps.append(desc)

    zipf = zipfile.ZipFile(zip_file)
    ot_caps = zipf.open("submit.txt").readlines()

    ot_caps = [smi.decode("utf-8").strip() for smi in ot_caps]

    assert len(truth_caps) + 134 + 3 == len(
        ot_caps
    ), "Different number of ground truth and predictions."  # 134 and 3 are FDA and SAB molecules

    print("Predicted and Reference SMILES lists read.")

    print("Calculating string metrics.")
    bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score = evaluate(
        truth_caps, ot_caps
    )

    scores = {}
    scores["BLEU-2"] = bleu2
    scores["BLEU-4"] = bleu4
    scores["Rouge-1"] = rouge_1
    scores["Rouge-2"] = rouge_2
    scores["Rouge-L"] = rouge_l
    scores["Meteor"] = _meteor_score

    print(scores)

    with open(os.path.join(score_dir, out_name + ".json"), "w") as score_file:
        score_file.write(json.dumps(scores))


sub_dir = "submissions/"
onlyfiles = [f for f in listdir(sub_dir) if isfile(join(sub_dir, f))]

for zip_file in onlyfiles:
    print(zip_file)
    create_scores(sub_dir + zip_file, out_name=zip_file.split(".")[0])
