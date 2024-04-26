# Evaluation Code: Translation between Molecules and Natural Language
Task evaluation code for "[L+M-24: Building a Dataset for Language+Molecules @ ACL 2024](https://arxiv.org/abs/2403.00791)".

Based on "[Translation between Molecules and Natural Language](https://arxiv.org/abs/2204.11817)".


## Installation
The requirements for the evaluation code conda environment are in environment_eval.yml. An environment can be created using the following commands: 

```
conda env create -n MolTextTranslationEval -f environment_eval.yml python=3.9
conda activate MolTextTranslationEval
python -m spacy download en_core_web_sm
pip install git+https://github.com/samoturk/mol2vec
python -c "import nltk; nltk.download('wordnet')"
chmod u+x *.sh
unzip ../additional_data.zip
```

Follow https://github.com/samoturk/mol2vec/issues/14 for mol2vec installion. A modified version of features.py for this can be found in the repo. 
An old version of FCD (1.1) is used because https://github.com/bioinf-jku/FCD/issues/14 isn't fixed. 

Certain additional data files are required for some commands (`nested_props.json`, `train_withheld_combos.txt`). These can be found in `../additional_data.zip`.

### Downloads

* [test_outputfinal_weights.320.pt](https://uofi.box.com/s/es16alnhzfy1hpagf55fu48k49f8n29x) should be placed in "evaluation/t2m_output".
It can be downloaded using ```curl -L  https://uofi.box.com/shared/static/es16alnhzfy1hpagf55fu48k49f8n29x --output test_outputfinal_weights.320.pt```

## Input format
The input format should be a tab-separated txt file with three columns and the header 'SMILES ground truth  output' for smiles2caption or 'description	ground truth	output' for caption2smiles. 

## Evaluation Commands

<table>
  <tr>
    <td>Code</td>
    <td>Evaluation</td>
  </tr>
  <tr>
    <td colspan="2">Evaluating SMILES to Caption</td>
  </tr>
  <tr>
    <td>python text_translation_metrics.py --input_file smiles2caption_example.txt</td>
    <td>Evaluate all NLG metrics.</td>
  </tr>
  <tr>
    <td>./text_text2mol_metric.sh smiles2caption_example.txt</td>
    <td>Evaluate Text2Mol metric for caption generation.</td>
  </tr>
  <tr>
    <td>./text_text2mol_metric_gt.sh smiles2caption_example.txt</td>
    <td>Evaluate Text2Mol metric for the ground truth.</td>
  </tr>
  <tr>
    <td>python text_property_metrics.py --input_file smiles2caption_example.txt --output_file tmp.txt</td>
    <td>Evaluate property metrics for caption generation.</td>
  </tr>
  <tr>
    <td colspan="2">Evaluating Caption to SMILES</td>
  </tr>
  <tr>
    <td>python mol_translation_metrics.py --input_file caption2smiles_example.txt</td>
    <td>Evaluate BLEU, Exact match, and Levenshtein metrics.</td>
  </tr>
  <tr>
    <td>python fingerprint_metrics.py --input_file caption2smiles_example.txt</td>
    <td>Evaluate fingerprint metrics.</td>
  </tr>
  <tr>
    <td>./mol_text2mol_metric.sh caption2smiles_example.txt</td>
    <td>Evaluate Text2Mol metric for molecule generation.</td>
  </tr>
  <tr>
    <td>./mol_text2mol_metric_gt.sh caption2smiles_example.txt</td>
    <td>Evaluate Text2Mol metric for the ground truth.</td>
  </tr>
  <tr>
    <td>python fcd_metric.py --input_file caption2smiles_example.txt</td>
    <td>Evaluate FCD metric for molecule generation.</td>
  </tr>
  <tr>
    <td>python create_heldout_file.py --input_file caption2smiles_example.txt --output_file caption2smiles_example_HO.txt --direction molecule</td>
    <td>Create a results file only containing results with held-out combo data points. File is currently hardcoded to use combos from the train-validation split. </td>
  </tr>
</table>



### Citation
If you found our work useful, please cite:


```bibtex
@article{edwards2024_LPM24,
  title={L+M-24: Building a Dataset for Language+Molecules @ ACL 2024},
  author={Edwards, Carl and Wang, Qingyun and Zhou, Lawrence and Ji, Heng},
  journal={arXiv preprint arXiv:2403.00791},
  year={2024}
}

@inproceedings{edwards-etal-2022-translation,
    title = "Translation between Molecules and Natural Language",
    author = "Edwards, Carl  and
      Lai, Tuan  and
      Ros, Kevin  and
      Honke, Garrett  and
      Cho, Kyunghyun  and
      Ji, Heng",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.26",
    pages = "375--413",
}
```
