'''
Code from https://github.com/blender-nlp/MolT5

```bibtex
@article{edwards2022translation,
  title={Translation between Molecules and Natural Language},
  author={Edwards, Carl and Lai, Tuan and Ros, Kevin and Honke, Garrett and Ji, Heng},
  journal={arXiv preprint arXiv:2204.11817},
  year={2022}
}
```
'''


import pickle
import argparse
import csv

import os.path as osp

import numpy as np

import torch

from text2mol.code.models import MLPModel

from transformers import BertTokenizerFast

from sklearn.metrics.pairwise import cosine_similarity

from rdkit import Chem

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

parser = argparse.ArgumentParser()

parser.add_argument('--use_gt', action=argparse.BooleanOptionalAction)

parser.add_argument('--input_file', type=str, default='smiles2caption_example.txt', help='path where test generations are saved')

parser.add_argument('--text_model', type=str, default='allenai/scibert_scivocab_uncased', help='Desired language model.')

parser.add_argument('--checkpoint', type=str, default='t2m_output/test_outputfinal_weights.320.pt', help='Text2Mol checkpoint to use.')

parser.add_argument('--text_trunc_length', type=str, default=256, help='tokenizer maximum length')


args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


mol2vec = {}

m2v_embs = 'tmp.csv'
with open(m2v_embs) as f:

    csvreader = csv.reader(f, delimiter=',')
    headers = next(csvreader)

    for i, row in enumerate(csvreader):
        
        if i % 1000 == 0: print(i)
    
        smi = row[3]

        mol_str = " ".join(row[-300:])

        mol2vec[smi] = np.fromstring(mol_str, sep = " ")


outputs = []

with open(osp.join(args.input_file)) as f:
    reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    for n, line in enumerate(reader):
        out_tmp = line['output'][6:] if line['output'].startswith('[CLS] ') else line['output']
        m = Chem.MolFromSmiles(line['SMILES'])
        smi = Chem.MolToSmiles(m)
        outputs.append((smi, line['ground truth'], out_tmp))


text_tokenizer = BertTokenizerFast.from_pretrained(args.text_model)

model = MLPModel(ninp = 768, nhid = 600, nout = 300)

tmp = model.to(device)

model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)

model.eval()

sims = []

mol_embs = []
text_embs = []

bad_mol = 0

with torch.no_grad():
    for i, (smi, gt, out) in enumerate(outputs):
        
        if i % 100 == 0: print(i, 'processed.')
        
        if args.use_gt: text = gt
        else: text = out

        try:
            m2v = mol2vec[smi]
        except:
            bad_mol += 1
            continue

        #print(text)
        text_input = text_tokenizer(text, truncation=True, max_length=args.text_trunc_length,
                                            padding='max_length', return_tensors = 'pt')

        input_ids = text_input['input_ids'].to(device)
        attention_mask = text_input['attention_mask'].to(device)
        molecule = torch.from_numpy(m2v).reshape((1,300)).to(device).float()

        text_emb, mol_emb = model(input_ids, molecule, attention_mask)

        text_emb = text_emb.cpu().numpy()
        mol_emb = mol_emb.cpu().numpy()

        text_embs.append(text_emb)
        mol_embs.append(mol_emb)

        sims.append(cosine_similarity(text_emb, mol_emb)[0][0])
        

print('Average Similarity:', np.mean(sims))

text_embs = np.array(text_embs).squeeze()
mol_embs = np.array(mol_embs).squeeze()

mat = cosine_similarity(text_embs, mol_embs)
print('Negative Similarity:', np.mean(mat[np.eye(mat.shape[0]) == 0]))
