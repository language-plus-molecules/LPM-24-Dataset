'''
Code from https://github.com/language-plus-molecules/LPM-24-Dataset

```bibtex
@article{edwards2024_LPM24,
  title={L+M-24: Building a Dataset for Language+Molecules @ ACL 2024},
  author={Edwards, Carl and Wang, Qingyun and Zhou, Lawrence and Ji, Heng},
  journal={arXiv preprint arXiv:},
  year={2024}
}
```
'''

import argparse
import csv

import os.path as osp

import numpy as np

from transformers import BertTokenizerFast

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

import json
from collections import defaultdict
import copy

from tqdm import tqdm

def flatten(dictionary, separator='__'):
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


def unflatten(dictionary, separator='__'):
    rv = {}
    for key in dictionary:
        if '__' in key:
            spl = key.split(separator)
            nested_set(rv, spl, dictionary[key])
        else:
            rv[key] = dictionary[key]

    return rv





def zero_division(n, d):
    return n / d if d else None


#https://stackoverflow.com/questions/3847386/how-to-test-if-a-list-contains-another-list-as-a-contiguous-subsequence
def contains(small, big):
    for i in range(len(big)-len(small)+1):
        for j in range(len(small)):
            if big[i+j] != small[j]:
                break
        else:
            return True
    return False




def evaluate(text_model, input_file, output_file, text_trunc_length):
    outputs = []

    with open(osp.join(input_file), encoding='utf8') as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for n, line in enumerate(reader):
            out_tmp = line['output'][6:] if line['output'].startswith('[CLS] ') else line['output']
            outputs.append((line['SMILES'], line['ground truth'], out_tmp))

    text_tokenizer = BertTokenizerFast.from_pretrained(text_model)

    def process(outputs, prop_list):
    
        TP = defaultdict(int)
        TN = defaultdict(int)
        FP = defaultdict(int)
        FN = defaultdict(int)

        prop_list_tok = [(text_tokenizer.tokenize(p.split('__')[-1].lower()), p) for p in prop_list]

        for smi, gt, out in tqdm(outputs):

            gtl = text_tokenizer.tokenize(gt.lower())
            outl = text_tokenizer.tokenize(out.lower())


            for pl, prop in prop_list_tok:
                
                gtc = contains(pl, gtl)
                outc = contains(pl, outl)


                if gtc and outc: TP[prop] += 1
                if gtc and not outc: FN[prop] += 1
                if not gtc and outc: FP[prop] += 1
                if not gtc and not outc: TN[prop] += 1

        return (TP, TN, FP, FN)

    nested_props = json.load(open('nested_props.json', encoding='utf-8'))
    
    flattened_props = flatten(nested_props, separator='__')

    TP, TN, FP, FN = process(outputs, flattened_props)

    precision = {}
    recall = {}
    f1 = {}
    accuracy = {}

    for fp in flattened_props:
        precision[fp] = zero_division(TP[fp], (TP[fp] + FP[fp]))
        recall[fp] = zero_division(TP[fp], (TP[fp] + FN[fp]))

        f1[fp] = zero_division((2*TP[fp]), (2*TP[fp] + FP[fp] + FN[fp]))
        accuracy[fp] = zero_division((TN[fp] + TP[fp]), (TN[fp] + TP[fp] + FN[fp] + FP[fp]))

    uf_acc = unflatten(accuracy)
    uf_prec = unflatten(precision)
    uf_recall = unflatten(recall)
    uf_f1 = unflatten(f1)


    def take_average(dictionary, f):

        for key in dictionary:
            if isinstance(dictionary[key], dict):
                dictionary[key] = take_average(dictionary[key], f=f)
                print(key, dictionary[key], file=f)
                return dictionary
            elif isinstance(dictionary[key], float):
                continue
            elif dictionary[key] == None:
                continue
            else:
                print(type(dictionary[key]))
                zz
        val_list = [dictionary[key] for key in dictionary if dictionary[key] != None]
        return np.mean(val_list) if len(val_list) != 0 else 0.0

    def process_combos(outputs, combos):
    
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        combos_tok = [(text_tokenizer.tokenize(p[0].split('__')[-1].lower()),
            text_tokenizer.tokenize(p[1].split('__')[-1].lower())) for p in combos]

        for smi, gt, out in tqdm(outputs):

            gtl = text_tokenizer.tokenize(gt.lower())
            outl = text_tokenizer.tokenize(out.lower())

            for (c1, c2), combo in zip(combos_tok, combos):

                gtc1 = contains(c1, gtl)
                outc1 = contains(c1, outl)
                
                gtc2 = contains(c2, gtl)
                outc2 = contains(c2, outl)
                
                if (gtc1 and gtc2) and (outc1 and outc2): TP+=1
                if (gtc1 and gtc2) and (not outc1 or not outc2): FN+=1
                if (not gtc1 or not gtc2) and (outc1 and outc2): FP+=1
                if (not gtc1 or not gtc2) and (not outc1 or outc2): TN+=1

        return (TP, TN, FP, FN)

    combos = [eval(line.strip().split('\t')[0]) for line in open('train_withheld_combos.txt').readlines()]

    TPc, TNc, FPc, FNc = process_combos(outputs, combos)


    with open(output_file, 'w', encoding='utf8') as f:
        

        print('Accuracy:', file=f)
        tmp_dict = copy.deepcopy(uf_acc)
        while True:
            tmp_dict = take_average(tmp_dict, f=f)

            if isinstance(tmp_dict, float): break
        print('Overall:', tmp_dict, file=f)

        print(file=f)
        print(file=f)
        
        print('Precision:', file=f)
        tmp_dict = copy.deepcopy(uf_prec)
        while True:
            tmp_dict = take_average(tmp_dict, f=f)

            if isinstance(tmp_dict, float): break
        print('Overall:', tmp_dict, file=f)

        print(file=f)
        print(file=f)
        
        print('Recall:', file=f)
        tmp_dict = copy.deepcopy(uf_recall)
        while True:
            tmp_dict = take_average(tmp_dict, f=f)

            if isinstance(tmp_dict, float): break
        print('Overall:', tmp_dict, file=f)

        print(file=f)
        print(file=f)
        
        print('F-1 Score:', file=f)
        tmp_dict = copy.deepcopy(uf_f1)
        while True:
            tmp_dict = take_average(tmp_dict, f=f)

            if isinstance(tmp_dict, float): break
        print('Overall:', tmp_dict, file=f)

        print(file=f)
        print(file=f)

        precision_comb = zero_division(TPc, (TPc + FPc))
        recall_comb = zero_division(TPc, (TPc + FNc))

        f1_comb = zero_division((2*TPc), (2*TPc + FPc + FNc))
        accuracy_comb = zero_division((TNc + TPc), (TNc + TPc + FNc + FPc))

        print('Held-Out Combo Metrics:', file=f)
        print('Acc:', accuracy_comb, file=f)
        print('Precision:', precision_comb, file=f)
        print('Recall:', recall_comb, file=f)
        print('F1:', f1_comb, file=f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_model', type=str, default='allenai/scibert_scivocab_uncased', help='Desired language model tokenizer.')
    parser.add_argument('--input_file', type=str, default='smiles2caption_example.txt', help='path where test generations are saved')
    parser.add_argument('--output_file', type=str, default='tmp.txt', help='path where output values are saved.')
    parser.add_argument('--text_trunc_length', type=str, default=512, help='tokenizer maximum length')
    args = parser.parse_args()
    evaluate(args.text_model, args.input_file, args.output_file, args.text_trunc_length)
