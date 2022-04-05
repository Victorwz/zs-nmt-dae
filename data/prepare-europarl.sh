export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_CTYPE=en_US.UTF-8
 
file=training-parallel-europarl-v7.tgz

if [ -f $file ]; then
    echo "$file already exists, skipping download"
else
    wget http://statmt.org/wmt13/training-parallel-europarl-v7.tgz
    tar zxvf $file
fi

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

srcpath=training
prep=EuroParl-raw-data/mnmt-dn
tmp=$prep/tmp

rm -rf $prep
mkdir -p $prep $tmp

SRCS=(
    "de-en"
    "fr-en"
)

PROC_SCRIPTS=mosesdecoder/scripts
TOKENIZER=$PROC_SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$PROC_SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$PROC_SCRIPTS/tokenizer/normalize-punctuation.perl
LC=$PROC_SCRIPTS/tokenizer/lowercase.perl
REM_NON_PRINT_CHAR=$PROC_SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000

echo "pre-processing train data..."
for SRC in "${SRCS[@]}"; do
    echo $SRC
    array=(${SRC//-/ })
    src=${array[0]}
    tgt=${array[1]}
    
    for LANG in ${array[@]}; do
        echo $LANG
        cat $srcpath/europarl-v7.$SRC.$LANG | \
            perl $NORM_PUNC $LANG | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $LANG | \
            perl $LC > $tmp/europarl.tokenized.$SRC.$LANG
    done

    python noising_plain_text.py --input-file $tmp/europarl.tokenized.$SRC.en \
            --output-file $tmp/europarl.denoising.$SRC.en \
            --denoising-mask-ratio 0.3 --denoising-replace-length 1 --denoising-mask-length span-poisson
    
done

TRAIN=$prep/train.mnmt
BPE_CODE=$prep/code
rm $TRAIN

cat $tmp/europarl.tokenized.* >> $TRAIN
cat $tmp/europarl.denoising.* >> $TRAIN

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for l in de fr; do 
    python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/europarl.tokenized.${l}-en.${l} > $tmp/europarl.bpe.tokenized.${l}-en.${l}
    python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/europarl.tokenized.${l}-en.en > $tmp/europarl.bpe.tokenized.${l}-en.en
    python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/europarl.denoising.${l}-en.en > $tmp/europarl.bpe.denoising.${l}-en.en

    awk '{print "<2en> "$0}' $tmp/europarl.bpe.tokenized.${l}-en.${l} >> $prep/train.src
    cat $tmp/europarl.bpe.tokenized.${l}-en.en >> $prep/train.tgt
    awk '{print "<2'$l'> "$0}' $tmp/europarl.bpe.tokenized.${l}-en.en >> $prep/train.src
    cat $tmp/europarl.bpe.tokenized.${l}-en.${l} >> $prep/train.tgt
    awk '{print "<2en> "$0}' $tmp/europarl.bpe.denoising.${l}-en.en >> $prep/train.src
    cat $tmp/europarl.bpe.tokenized.${l}-en.en >> $prep/train.tgt
done

perl $CLEAN -ratio 1.5 $prep/train src tgt $prep/train.clean 1 250

# Tokenize and encode devset and testset

dev_test_file=devsets.tgz
if [ -f $dev_test_file ]; then
    echo "$dev_test_file already exists, skipping download"
else
    wget https://www.statmt.org/wmt07/devsets.tgz
    tar zxvf $dev_test_file -C $srcpath
fi

for l in de fr en; do
    cat $srcpath/wmt07/dev/dev2006.$l | \
        perl $NORM_PUNC $l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads 8 -a -l $l | \
        perl $LC > $tmp/valid.$l

    cat $srcpath/wmt07/devtest/devtest2006.$l | \
        perl $NORM_PUNC $l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads 8 -a -l $l | \
        perl $LC > $tmp/test.$l

    python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/valid.$l > $tmp/valid.bpe.$l
    python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/test.$l > $tmp/test.bpe.$l
done

# Concatenating the full validation set on two zero-shot directions

SRCS=(
    "de-fr"
    "fr-de"
)

echo "pre-processing valid data..."
for SRC in "${SRCS[@]}"; do
    echo $SRC
    array=(${SRC//-/ })
    src=${array[0]}
    tgt=${array[1]} 
    echo $src,$tgt

    awk '{print "<2'$tgt'> "$0}' $tmp/valid.bpe.$src >> $prep/valid.src

    cat $tmp/valid.bpe.$tgt >> $prep/valid.tgt
done
