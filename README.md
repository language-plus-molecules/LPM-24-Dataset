# *L+M-24* Dataset
This repository contains information on the creation, evaluation, and benchmark models for the L+M-24 Dataset. L+M-24 will be featured as the shared task at The Language + Molecules Workshop at ACL 2024.

-----------------------------------------

Language-molecule models have emerged as an exciting direction for molecular discovery and understanding. However, training these models is challenging due to the scarcity of molecule-language pair datasets. At this point, datasets have been released which are 1) small and scraped from existing databases, 2) large but noisy and constructed by performing entity linking on the scientific literature, and 3) template-based built on prediction datasets. In this document, we detail the *L+M-24* dataset, which has been created for the Language + Molecules Workshop shared task at ACL 2024. In particular, *L+M-24* is designed to focus on three key benefits of natural language in molecule design: compositionality, functionality, and abstraction.

Please see the manuscript for this dataset [here](https://arxiv.org/pdf/2403.00791.pdf).

-----------------------------------------

## News

* The presentation on results of the shared task is [now released](/shared_task.pdf). 
* Final evaluation code and shared task submissions have been made available [here](/final_evaluation/). The submissions are being made available to allow the study of ensemble methods. 
  * Details on the mystery molecules used in the shared task are [available](/shared_task.pdf). Our majority voting ensemble for the [mystery molecules](/final_evaluation/captioning/mystery_mols.txt) is available [here](/final_evaluation/captioning/mystery_molecules_ensemble.txt). 

* The [official leaderboard](https://language-plus-molecules.github.io/#leaderboard) is now available!! 

* Submissions can still be uploaded to Codabench! See the competitions at: [Molecule Captioning](https://www.codabench.org/competitions/2914) and [Molecule Generation](https://www.codabench.org/competitions/3014). See instructions on the [website](https://language-plus-molecules.github.io/#submission).

  * Example MolT5-Small submission files are available as "MolT5-Small_cap2smi_submit.zip" and "MolT5-Small_smi2cap_submit.zip". 

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
    <td><a href="https://huggingface.co/datasets/language-plus-molecules/LPM-24_eval-caption"> LPM-24_eval-caption </a></td>
    <td>The evaluation data for molecule caption generation. </td>
  </tr>
</table>

Further, datasets are available in zipped file `data.zip`. Some files that may be useful for training or necessary for evaluation are contained in `additional_data.zip`. 

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


