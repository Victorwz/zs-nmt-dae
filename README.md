# zs-nmt-dae
Official implementation of EMNLP 2021 Paper "Rethinking Zero-shot Neural Machine Translation: From a Perspective of Latent Variables". 

## Citation
Please cite our paper if you find this repository helpful in your research:
```
comming soon
```

## Requirements and Installation
* Python version == 3.6
* [PyTorch](http://pytorch.org/) version == 1.6.0
* sacremoses == 
* sacrebleu == 
* **To install fairseq**:
```
git clone https://github.com/Victorwz/zs-nmt-dae.git;
cd zs-nmt-dae;
pip install --editable ./;
```

## Data Downloading
We conduct experiments on two multilingual corpus [MultiUN](https://conferences.unite.un.org/uncorpus) and [Europarl](http://www.statmt.org/europarl/).

For downloading MultiUN, please refer to its [official website and scripts](https://conferences.unite.un.org/UNCORPUS/en/DownloadOverview). The downloaded corpus should be put in the folder ./data/MultiUN. Or you can use our script to download the corpus:
```
cd data;
bash download-multiun.sh
```

For downloading Europarl, please refer to its [official website and scripts](http://www.statmt.org/europarl/). The official validation and test sets are [WMT devtest2006 and testset2006](http://matrix.statmt.org/test_sets/list). The downloading script might be too complicated and we highly suggest that you could download manually.

## Data Preprocessing
For preprocess MultiUN, please run the following shell scripts:
```
cd data;
bash prepare-multiun.sh
```

## Binalizing and Training with FairSeq
