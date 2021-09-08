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
git clone https://github.com/pytorch/fairseq;
cd fairseq;
pip install --editable ./;
```

## Data Downloading
We conduct experiments on two multilingual corpus [MultiUN](https://conferences.unite.un.org/uncorpus) and [Europarl]

For downloading MultiUN, please refer to its [official website and scripts](https://conferences.unite.un.org/UNCORPUS/en/DownloadOverview). The downloaded corpus should be put in the folder ./data/MultiUN.



## Data Pre-processing
The scripts for data pre-processing is available at 