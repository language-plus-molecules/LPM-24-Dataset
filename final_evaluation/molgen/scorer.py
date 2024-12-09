print("In the scoring function")

import sys
import os

from os import listdir

import pickle
import argparse
import csv
import json


import numpy as np

# load metric stuff

from nltk.translate.bleu_score import corpus_bleu

from Levenshtein import distance as lev

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem

import io
import zipfile

from os import listdir
from os.path import isfile, join

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

print("Imported packages")


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


def create_scores(zip_file, out_name):
    score_dir = "scores"

    truth_file = "eval-molgen.txt"
    truth = open(truth_file).readlines()
    truth_smis = []
    for line in truth:
        smi, desc, _ = line.split("\t")
        truth_smis.append(smi)

    zipf = zipfile.ZipFile(zip_file)
    ot_smis = zipf.open("submit.txt").readlines()

    ot_smis = [smi.decode("utf-8").strip() for smi in ot_smis]

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
    scores["BLEU"] = bleu_score
    scores["exact_match"] = exact_match_score
    scores["levenshtein"] = levenshtein_score
    scores["validity"] = validity_score
    scores["maccs_sim"] = maccs_sims_score
    scores["rdk_sim"] = rdk_sims_score
    scores["morgan_sim"] = morgan_sims_score
    scores["FCD"] = fcd_score

    with open(os.path.join(score_dir, out_name + ".json"), "w") as score_file:
        score_file.write(json.dumps(scores))


sub_dir = "submissions/"
onlyfiles = [f for f in listdir(sub_dir) if isfile(join(sub_dir, f))]

for zip_file in onlyfiles:
    print(zip_file)
    create_scores(sub_dir + zip_file, out_name=zip_file.split(".")[0])
