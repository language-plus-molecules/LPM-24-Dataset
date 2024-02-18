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


from rdkit import Chem

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', type=str, default='caption2smiles_example.txt', help='path where test generations are saved')
parser.add_argument('--direction', type=str, default='molecule', help="'molecule' for cap2smi, 'caption' for smi2cap")
parser.add_argument('--use_gt', action=argparse.BooleanOptionalAction)


args = parser.parse_args()

outputs = []

bad_mols = 0

with open(osp.join(args.input_file)) as f:
    reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    for n, line in enumerate(reader):
        try:
            if args.direction == 'molecule':
                gt_smi = line['ground truth']
                ot_smi = line['output']
                desc = line['description']

                gt_m = Chem.MolFromSmiles(gt_smi)
                gt_smi = Chem.MolToSmiles(gt_m)
                
                if args.use_gt:
                    outputs.append((desc, gt_smi, gt_m))

                m = Chem.MolFromSmiles(ot_smi)
                ot_smi = Chem.MolToSmiles(m)


                if ot_smi == '': #fixes a downstream error in mol2vec
                    raise ValueError('Empty molecule.')

                if not args.use_gt:
                    outputs.append((desc, gt_smi, m))

            else:
                gt_smi = line['SMILES']
                m = Chem.MolFromSmiles(gt_smi)
                gt_smi = Chem.MolToSmiles(m)
            
                desc = line['ground truth']

                outputs.append((desc, gt_smi, m))
        except:
            bad_mols += 1
if not args.use_gt: print('validity:', len(outputs)/(len(outputs)+bad_mols))

with Chem.SDWriter('tmp.sdf') as w:
    for o in outputs:
        m = o[2]
        m.SetProp("GT_SMI", o[1])
        w.write(m)


