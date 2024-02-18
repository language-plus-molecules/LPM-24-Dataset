# *L+M-24* Dataset
This repository contains information on the creation, evaluation, and benchmark models for the L+M-24 Dataset. L+M-24 will be featured as the shared task at The Language + Molecules Workshop at ACL 2024.

-----------------------------------------

Language-molecule models have emerged as an exciting direction for molecular discovery and understanding. However, training these models is challenging due to the scarcity of molecule-language pair datasets. At this point, datasets have been released which are 1) small and scraped from existing databases, 2) large but noisy and constructed by performing entity linking on the scientific literature, and 3) template-based built on prediction datasets. In this document, we detail the *L+M-24* dataset, which has been created for the Language + Molecules Workshop shared task at ACL 2024. In particular, *L+M-24* is designed to focus on three key benefits of natural language in molecule design: compositionality, functionality, and abstraction.

Please see the manuscript for this dataset [here](https://blender.cs.illinois.edu/paper/molecules24.pdf).

-----------------------------------------

## Dataset Download

Datasets are made available to download through HuggingFace datasets. 

<table>
  <tr>
    <td>Split</td>
    <td>Link</td>
    <td>Description</td>
  </tr>
  <tr>
    <td>Train</td>
    <td><a href="https://huggingface.co/datasets/language-plus-molecules/LPM-24_train"> LPM-24_train </a></td>
    <td>The full training data for the shared task.</td>
  </tr>
  <tr>
    <td>Train-Extra</td>
    <td> <a href="https://huggingface.co/datasets/language-plus-molecules/LPM-24_train-extra"> LPM-24_train-extra </a> </td>
    <td>Extra training data for the shared task with 5 captions generated for each molecule.</td>
  </tr>
  <tr>
    <td>Evaluation -- Molecule Generation</td>
    <td><a href="https://huggingface.co/datasets/language-plus-molecules/LPM-24_eval-molgen"> LPM-24_eval-molgen </a></td>
    <td>The evaluation data for molecule generation. Only input captions are included.</td>
  </tr>
  <tr>
    <td>Evaluation -- Caption Generation</td>
    <td>LPM-24_eval-caption</td>
    <td>The evaluation data for molecule caption generation. This split isn't available yet because we're still adding to it!</td>
  </tr>
</table>

Further, datasets are available in zipped file `data.zip'. Some files that may be useful for training or necessary for evaluation are contained in `additional_data.zip'. 

------------------------------------
## Evaluation

Evaluation code and instructions can be found in [evaluation](/evaluation).


------------------------------------


## Source Datasets

We would like to thank the input databases we used to construct this dataset!

* [Chemical Function (CheF)](https://chefdb.app/)
  * [paper](https://arxiv.org/abs/2309.08765)
* [ChemFOnt: the chemical functional ontology resource](https://www.chemfont.ca/)
  * [paper](https://academic.oup.com/nar/article/51/D1/D1220/6777791)
* [Pubchem](https://pubchem.ncbi.nlm.nih.gov/)
  * [paper](https://academic.oup.com/nar/article/47/D1/D1102/5146201)


### Citation
If you found this dataset or code useful, please cite:


```bibtex
@article{edwards2024_LPM24,
  title={L+M-24: Building a Dataset for Language+Molecules @ ACL 2024},
  author={Edwards, Carl and Wang, Qingyun and Zhou, Lawrence and Ji, Heng},
  journal={arXiv preprint arXiv:},
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

as well as the source datasets. 


