print("In the scoring function")

import sys
import os

from os import listdir

import pickle
import argparse
import csv
import json

from tqdm import tqdm

import numpy as np

import random

# load metric stuff

from nltk.translate.bleu_score import corpus_bleu

# from nltk.translate.meteor_score import meteor_score

from Levenshtein import distance as lev

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem

from transformers import BertTokenizerFast

import io
import zipfile

from os import listdir
from os.path import isfile, join

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

print("Imported packages")


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


def process_combos(descs, combos, text_model="allenai/scibert_scivocab_uncased"):

    text_tokenizer = BertTokenizerFast.from_pretrained(text_model)

    combos_tok = [
        (
            text_tokenizer.tokenize(p[0].split("__")[-1].lower()),
            text_tokenizer.tokenize(p[1].split("__")[-1].lower()),
        )
        for p in combos
    ]

    keep_lines = []
    for desc in tqdm(descs):

        gtl = text_tokenizer.tokenize(desc.lower())

        final_flag = False

        for (c1, c2), combo in zip(combos_tok, combos):

            gtc1 = contains(c1, gtl)
            gtc2 = contains(c2, gtl)
            if gtc1 and gtc2:
                final_flag = True

        keep_lines.append(final_flag)

    return keep_lines


def get_mask(descs):
    combos = [
        eval(line.strip().split("\t")[0])
        for line in open("withheld_combos.txt").readlines()
    ]

    mask = process_combos(descs, combos)

    return mask


def evaluate_fp(gt_smis, ot_smis, morgan_r=2, verbose=False):
    outputs = []
    bad_mols = 0

    valid_smiles = []

    for n, (gt_smi, ot_smi) in enumerate(zip(gt_smis, ot_smis)):
        try:
            gt_m = Chem.MolFromSmiles(gt_smi)
            ot_m = Chem.MolFromSmiles(ot_smi)

            if ot_m == None:
                raise ValueError("Bad SMILES")
            else:
                valid_smiles.append(ot_smi)
            outputs.append((gt_m, ot_m))
        except:
            bad_mols += 1
            outputs.append((gt_m, None))
    validity_score = (len(outputs) - bad_mols) / (len(outputs))
    uniqueness_score = len(set(valid_smiles)) / (len(outputs) - bad_mols)
    if verbose:
        print("validity:", validity_score)
        print("uniqueness:", uniqueness_score)

    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []

    enum_list = outputs

    for i, (gt_m, ot_m) in enumerate(enum_list):

        if i % 100 == 0:
            if verbose:
                print(i, "processed.")

        if ot_m != None:
            MACCS_sims.append(
                DataStructs.FingerprintSimilarity(
                    MACCSkeys.GenMACCSKeys(gt_m),
                    MACCSkeys.GenMACCSKeys(ot_m),
                    metric=DataStructs.TanimotoSimilarity,
                )
            )
            RDK_sims.append(
                DataStructs.FingerprintSimilarity(
                    Chem.RDKFingerprint(gt_m),
                    Chem.RDKFingerprint(ot_m),
                    metric=DataStructs.TanimotoSimilarity,
                )
            )
            morgan_sims.append(
                DataStructs.TanimotoSimilarity(
                    AllChem.GetMorganFingerprint(gt_m, morgan_r),
                    AllChem.GetMorganFingerprint(ot_m, morgan_r),
                )
            )
        else:
            MACCS_sims.append(0)
            RDK_sims.append(0)
            morgan_sims.append(0)

    maccs_sims_score = np.mean(MACCS_sims)
    rdk_sims_score = np.mean(RDK_sims)
    morgan_sims_score = np.mean(morgan_sims)
    if verbose:
        print("Average MACCS Similarity:", maccs_sims_score)
        print("Average RDK Similarity:", rdk_sims_score)
        print("Average Morgan Similarity:", morgan_sims_score)
    return (
        validity_score,
        maccs_sims_score,
        rdk_sims_score,
        morgan_sims_score,
        uniqueness_score,
    )


def evaluate(gt_smi, ot_smi, verbose=False):
    outputs = [(a, b) for a, b in zip(gt_smi, ot_smi)]

    bleu_scores = []

    references = []
    hypotheses = []

    for i, (gt, out) in enumerate(outputs):

        if i % 100 == 0:
            if verbose:
                print(i, "processed.")

        gt_tokens = [c for c in gt]

        out_tokens = [c for c in out]

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

    # BLEU score
    bleu_score = corpus_bleu(references, hypotheses)
    if verbose:
        print("BLEU score:", bleu_score)
    rouge_scores = []

    references = []
    hypotheses = []

    levs = []

    num_exact = 0

    bad_mols = 0

    for i, (gt, out) in enumerate(outputs):

        hypotheses.append(out)
        references.append(gt)

        try:
            m_out = Chem.MolFromSmiles(out)
            m_gt = Chem.MolFromSmiles(gt)

            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt):
                num_exact += 1
        except:
            bad_mols += 1

        levs.append(lev(out, gt))

    # Exact matching score
    exact_match_score = num_exact / (i + 1)
    if verbose:
        print("Exact Match:")
        print(exact_match_score)

    # Levenshtein score
    levenshtein_score = np.mean(levs)
    if verbose:
        print("Levenshtein:")
        print(levenshtein_score)

    validity_score = 1 - bad_mols / len(outputs)
    if verbose:
        print("validity:", validity_score)

    return bleu_score, exact_match_score, levenshtein_score, validity_score


from fcd import get_fcd, load_ref_model, canonical_smiles


def evaluate_fcd(gt_smis, ot_smis, verbose=False):

    model = load_ref_model()

    canon_gt_smis = [w for w in canonical_smiles(gt_smis) if w is not None]
    canon_ot_smis = [w for w in canonical_smiles(ot_smis) if w is not None]

    canon_ot_smis = [smi for smi in canon_ot_smis if smi != ""]

    try:
        fcd_sim_score = get_fcd(canon_gt_smis, canon_ot_smis, model)
    except ValueError:
        fcd_sim_score = np.nan

    if verbose:
        print("FCD Similarity:", fcd_sim_score)

    return fcd_sim_score


score_dir = "scores_combo/"


def create_scores(zip_file, out_name):

    truth_file = "eval-molgen.txt"
    truth = open(truth_file).readlines()
    truth_smis = []
    descs = []
    for line in truth:
        smi, desc, _ = line.split("\t")
        truth_smis.append(smi)
        descs.append(desc)

    mask = get_mask(descs)

    zipf = zipfile.ZipFile(zip_file)
    ot_smis = zipf.open("submit.txt").readlines()

    ot_smis = [smi.decode("utf-8").strip() for smi in ot_smis]

    truth_smis = [smi for smi, flag in zip(truth_smis, mask) if flag]
    ot_smis = [smi for smi, flag in zip(ot_smis, mask) if flag]

    assert len(truth_smis) == len(
        ot_smis
    ), "Different number of ground truth and predictions."

    print("Predicted and Reference SMILES lists read.")

    print("Calculating string metrics.")
    bleu_score, exact_match_score, levenshtein_score, validity_score = evaluate(
        truth_smis, ot_smis, verbose=False
    )

    print("String metrics calculated.")
    print("Calculating fingerprint metrics.")

    (
        validity_score,
        maccs_sims_score,
        rdk_sims_score,
        morgan_sims_score,
        uniqueness_score,
    ) = evaluate_fp(truth_smis, ot_smis)
    print("Fingerprint metrics calculated.")

    print("Calculating FCD metric.")
    fcd_score = evaluate_fcd(truth_smis, ot_smis)
    print("Calculated FCD metric.")

    scores = {}
    scores["Combo BLEU"] = bleu_score
    scores["Combo exact_match"] = exact_match_score
    scores["Combo levenshtein"] = levenshtein_score
    scores["Combo validity"] = validity_score
    scores["Combo maccs_sim"] = maccs_sims_score
    scores["Combo rdk_sim"] = rdk_sims_score
    scores["Combo morgan_sim"] = morgan_sims_score
    scores["Combo FCD"] = fcd_score

    with open(os.path.join(score_dir, out_name + ".json"), "w") as score_file:
        score_file.write(json.dumps(scores))


sub_dir = "submissions/"
onlyfiles = [f for f in listdir(sub_dir) if isfile(join(sub_dir, f))]

random.shuffle(onlyfiles)

for zip_file in onlyfiles:
    print(zip_file)
    sys.stdout.flush()
    if os.path.exists(join(score_dir, zip_file.split(".")[0] + ".json")):
        continue

    create_scores(sub_dir + zip_file, out_name=zip_file.split(".")[0])
