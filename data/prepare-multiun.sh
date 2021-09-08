echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

srcpath=MultiUN
prep=MultiUN-raw-data-2M/mnmt-dn
tmp=$prep/tmp

rm -rf $prep
mkdir -p $prep $tmp

# The downloaded corpus has different naming pattern: en-zh en-ru ar-en
paste -d'|' $srcpath/en-zh/UNv1.0.en-zh.en $srcpath/en-zh/UNv1.0.en-zh.zh | cat -n | shuf -n 2000000 | sort -n | cut -f2 > $srcpath/en-zh/sample
cut -d'|' -f1 $srcpath/en-zh/sample > $tmp/multiun.zh-en.en
cut -d'|' -f2 $srcpath/en-zh/sample > $tmp/multiun.zh-en.zh

paste -d'|' $srcpath/en-ru/UNv1.0.en-ru.en $srcpath/en-ru/UNv1.0.en-ru.ru | cat -n | shuf -n 2000000 | sort -n | cut -f2 > $srcpath/en-ru/sample
cut -d'|' -f1 $srcpath/en-ru/sample > $tmp/multiun.ru-en.en
cut -d'|' -f2 $srcpath/en-ru/sample > $tmp/multiun.ru-en.ru

paste -d'|' $srcpath/ar-en/UNv1.0.ar-en.en $srcpath/ar-en/UNv1.0.ar-en.ar | cat -n | shuf -n 2000000 | sort -n | cut -f2 > $srcpath/ar-en/sample
cut -d'|' -f1 $srcpath/ar-en/sample > $tmp/multiun.ar-en.en
cut -d'|' -f2 $srcpath/ar-en/sample > $tmp/multiun.ar-en.ar

SRCS=(
    "ar-en"
    "ru-en"
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
# for SRC in "${SRCS[@]}"; do
for SRC in $SRCS; do
    echo $SRC
    array=(${SRC//-/ })
    src=${array[0]}
    tgt=${array[1]}
    
    for LANG in ${array[@]}; do
        echo $LANG
        cat $tmp/multiun.$SRC.$LANG | \
            perl $NORM_PUNC $LANG | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $LANG > $tmp/multiun.tokenized.$SRC.$LANG
    done

    python noising_plain_text.py --input-file $tmp/multiun.tokenized.$SRC.en \
            --output-file $tmp/multiun.denoising.$SRC.en \
            --denoising-mask-ratio 0.3 --denoising-replace-length 1 --denoising-mask-length span-poisson
    
done

python -m jieba $tmp/multiun.zh-en.zh -d > $tmp/multiun.tokenized.zh-en.zh

cat $tmp/multiun.zh-en.en | \
    perl $NORM_PUNC en | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -a -l en > $tmp/multiun.tokenized.zh-en.en

python noising_plain_text.py --input-file $tmp/multiun.tokenized.zh-en.en \
    --output-file $tmp/multiun.denoising.zh-en.en \
    --denoising-mask-ratio 0.3 --denoising-replace-length 1 --denoising-mask-length span-poisson


TRAIN=$prep/train.mnmt
BPE_CODE=$prep/code
rm $TRAIN

cat $prep/multiun.tokenized.* >> $TRAIN
cat $prep/multiun.denoising.* | shuf -n 2000000 >> $TRAIN

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for l in ar zh ru; do 
    python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/multiun.tokenized.${l}-en.${l} > $tmp/multiun.bpe.tokenized.${l}-en.${l}
    python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/multiun.tokenized.${l}-en.en > $tmp/multiun.bpe.tokenized.${l}-en.en
    python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/multiun.denoising.${l}-en.en > $tmp/multiun.bpe.denoising.${l}-en.en

    awk '{print "<2en> "$0}' $tmp/multiun.bpe.tokenized.${l}-en.${l} >> $prep/train.src
    cat $tmp/multiun.bpe.tokenized.${l}-en.en >> $prep/train.tgt
    awk '{print "<2'$l'> "$0}' $tmp/multiun.bpe.tokenized.${l}-en.en >> $prep/train.src
    cat $tmp/multiun.bpe.tokenized.${l}-en.${l} >> $prep/train.tgt
    awk '{print "<2en> "$0}' $tmp/multiun.bpe.noisy.${l}-en.en >> $prep/train.src
    cat $tmp/multiun.bpe.tokenized.${l}-en.en >> $prep/train.tgt
done

perl $CLEAN -ratio 1.5 $prep/train src tgt $prep/train.clean 1 250

# Tokenize and encode devset and testset

for l in ar ru en; do
    cat $srcpath/testsets/devset/UNv1.0.devset.$l | \
        perl $NORM_PUNC $l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads 8 -a -l $LANG > $tmp/valid.$l

    cat $srcpath/testsets/testset/UNv1.0.testset.$l | \
        perl $NORM_PUNC $l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads 8 -a -l $LANG > $tmp/test.$l

    python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/valid.$l > $tmp/valid.bpe.$l
    python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/test.$l > $tmp/test.bpe.$l
done

# Deal with Chinese tokenization with Jieba

python -m jieba $srcpath/testsets/devset/UNv1.0.devset.zh -d > $tmp/valid.zh
python -m jieba $srcpath/testsets/testset/UNv1.0.testset.zh -d > $tmp/test.zh

python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/valid.zh > $tmp/valid.bpe.zh
python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/test.zh > $tmp/test.bpe.zh

# Concatenating the full validation set on six zero-shot directions would build up a too-large validation corpus,
# leading to low efficiency during validation.
# We sub-sample 1k sentences from every zero-shot direction to build up a 6k-size validation set.

SRCS=(
    "ru-zh"
    "ru-ar"
    "zh-ru"
    "zh-ar"
    "ar-zh"
    "ar-ru"
)

echo "pre-processing valid data..."
for SRC in "${SRCS[@]}"; do
    echo $SRC
    array=(${SRC//-/ })
    src=${array[0]}
    tgt=${array[1]} 
    echo $src,$tgt

    awk '{print "<2'$tgt'> "$0}' $tmp/valid.bpe.$src | head -n 1000 >> $prep/valid.src

    cat $tmp/valid.bpe.$tgt | head -n 1000 >> $prep/valid.tgt
done