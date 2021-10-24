# zs-nmt-dae
Official implementation of EMNLP 2021 Paper "[Rethinking Zero-shot Neural Machine Translation: From a Perspective of Latent Variables](https://arxiv.org/abs/2109.04705)". 

## Citation
Please cite our paper if you find this repository helpful in your research:
```
@article{wang2021rethinking,
  title={Rethinking Zero-shot Neural Machine Translation: From a Perspective of Latent Variables},
  author={Wang, Weizhi and Zhang, Zhirui and Du, Yichao and Chen, Boxing and Xie, Jun and Luo, Weihua},
  journal={arXiv preprint arXiv:2109.04705},
  year={2021}
}
```

## Requirements and Installation
* Python version == 3.6
* [PyTorch](http://pytorch.org/) version == 1.5.0
* numpy == 1.19.5
* sacremoses == 0.0.43
* sacrebleu == 1.5.1
* jieba == 0.42.1
* tqdm == 4.59.0
* **To install revised fairseq 0.10.1**:
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

For downloading Europarl, please refer to its [official website and scripts](http://www.statmt.org/europarl/). The official validation and test sets of Europarl are [WMT devtest2006 and testset2006](http://matrix.statmt.org/test_sets/list). The downloading script might be too complicated and we highly suggest that you could download manually.

## Data Preprocessing
For preprocess MultiUN corpus, please run the following shell scripts:
```
cd data;
bash prepare-multiun.sh
```

## Binalizing and Training with FairSeq
For training multilingual NMT model with denoising autoencoder objective on MultiUN, please run the following shell scripts:
```
bash train_multiun_mnmt_dn.sh
```

## Decoding and Testing
For the decoding and testing on MultiUN, you need to first train the transformer model from scratch to get your checkpoint. Or you can use our checkpoint for reproducing the reported results in our paper.
The checkpoint, dictionary, and BPE code are available at [Google Drive](https://drive.google.com/file/d/1iLTJoV9tTAzk7U3RgYW5H9F9CbZycXvt/view?usp=sharing). You can download it and unzip to ./checkpoints/multiun_mnmt_denoising. You need to modify the model and dictionary path in testing script to run the script.

For testing, please run the following shell scripts:
```
bash test_multiun_mnmt_dn.sh
```

We also find that deploying the trick of averaging the last 5 checkpoints starting from checkpoint_valid_bleu_best may lead to a better performance. You can uncomment some part of the code in our scripts to test the functionality of this trick. However, we did not deploy this trick in our method and all baseline methods in our paper.

## Credit
Our project is developed based on [FairSeq](https://github.com/pytorch/fairseq) toolkit.
