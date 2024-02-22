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
            return True#i, i+len(small)
    return False




def process(text_model, input_file, output_file, text_trunc_length, direction):
    outputs = []

    with open(osp.join(input_file), encoding='utf8') as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for n, line in enumerate(reader):
            if direction == 'caption':
                out_tmp = line['output'][6:] if line['output'].startswith('[CLS] ') else line['output']
                outputs.append((line['SMILES'], line['ground truth'], out_tmp, line['SMILES'] + '\t' + line['ground truth'] + '\t' + out_tmp + '\n'))
            elif direction == 'molecule':
                outputs.append((line['description'], line['ground truth'], line['output'], line['description'] + '\t' + line['ground truth'] + '\t' + line['output'] + '\n'))

    text_tokenizer = BertTokenizerFast.from_pretrained(text_model)


    def process_combos(outputs, combos):
    

        combos_tok = [(text_tokenizer.tokenize(p[0].split('__')[-1].lower()),
            text_tokenizer.tokenize(p[1].split('__')[-1].lower())) for p in combos]

        keep_lines = []

        if direction == 'caption':
            for smi, gt, out, line in tqdm(outputs):

                gtl = text_tokenizer.tokenize(gt.lower())

                for (c1, c2), combo in zip(combos_tok, combos):

                    gtc1 = contains(c1, gtl)                
                    gtc2 = contains(c2, gtl)
                    if gtc1 and gtc2: 
                        keep_lines.append(line)
        elif direction == 'molecule':
            for desc, gt, out, line in tqdm(outputs):

                gtl = text_tokenizer.tokenize(desc.lower())

                for (c1, c2), combo in zip(combos_tok, combos):

                    gtc1 = contains(c1, gtl)                
                    gtc2 = contains(c2, gtl)
                    if gtc1 and gtc2: 
                        keep_lines.append(line)

        return keep_lines


    combos = [eval(line.strip().split('\t')[0]) for line in open('train_withheld_combos.txt').readlines()]

    lines = set(process_combos(outputs, combos))

    print(len(lines))

    with open(output_file, 'w', encoding='utf8') as f:

        if direction == 'caption':
            f.write('SMILES\tground truth\toutput\n')
        elif direction == 'molecule':
            f.write('description\tground truth\toutput\n')

        for line in lines:
            f.write(line)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_model', type=str, default='allenai/scibert_scivocab_uncased', help='Desired language model tokenizer.')
    parser.add_argument('--input_file', type=str, default='smiles2caption_example.txt', help='path where test generations are saved')
    parser.add_argument('--output_file', type=str, default='tmp0.txt', help='path where output values are saved.')
    parser.add_argument('--text_trunc_length', type=str, default=512, help='tokenizer maximum length')
    parser.add_argument('--direction', default='molecule', type=str,
                    help="'molecule' for cap2smi, 'caption' for smi2cap")

    args = parser.parse_args()
    process(args.text_model, args.input_file, args.output_file, args.text_trunc_length, args.direction)


