# Final Evaluation Code: L+M-24 Shared Task @ ACL 2024
Task evaluation code for "[L+M-24: Building a Dataset for Language+Molecules @ ACL 2024](https://arxiv.org/abs/2403.00791)".

Based on "[Translation between Molecules and Natural Language](https://arxiv.org/abs/2204.11817)".

## Studying Ensemble Methods

The released [captioning](./captioning/submissions/) and [molecule generation](./molgen/submissions) submissions can be used to study ensemble methods. In particular, the [extracted properties for molecule captioning](./captioning/prop_files.zip) may be useful for this.

## Installation
The requirements for the evaluation code conda environment are located in the [evaluation README](../evaluation/README.md).

## Input format
The input format should contain the output of your model with one output on each line. It should follow the order of the online dataset: huggingface.co/datasets/language-plus-molecules. Your input file should be renamed to `submit.txt` and then zipped and put in the submissions subdirectory. 

## Evaluation Commands

Please see the `run.sh` files to see how score files are prepared. Then run `combine_eval.py` to create a combined score. Run `print_csv.py` to print a leaderboard. 


### Citation
If you found our work useful, please cite:


```bibtex
@inproceedings{edwards2024_LPM24,
    title = "{L}+{M}-24: Building a Dataset for {L}anguage+{M}olecules @ {ACL} 2024",
    author = "Edwards, Carl  and
      Wang, Qingyun  and
      Zhao, Lawrence  and
      Ji, Heng",
    booktitle = "Proceedings of the 1st Workshop on Language + Molecules (L+M 2024)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.langmol-1.1",
    doi = "10.18653/v1/2024.langmol-1.1",
    pages = "1--9",
}

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
