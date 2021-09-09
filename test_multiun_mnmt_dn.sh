TEXT=data/MultiUN-raw-data-2M

# Please download our checkpoint, dictionary to reprocude the results in our paper
MODEL=multiun_mnmt_denoising
MODEL_PATH=checkpoints/multiun_mnmt_denoising/checkpoint_best.pt
DICT=data-bin/multiun_mnmt_denoising/dict.src.txt

LC=data/mosesdecoder/scripts/tokenizer/lowercase.perl
DETOK=data/mosesdecoder/scripts/tokenizer/detokenizer.perl

TEST=$TEXT/${MODEL}_tests
TMP=$TEXT/tmp
rm -rf $TEST
mkdir $TEST

rm -rf data-bin/${MODEL}_tests

# supervised directions
# SRCS=(
#     "ar-en"
#     "en-ar"
#     "zh-en"
#     "en-zh"
#     "ru-en"
#     "en-ru"
# )

# zero-shot directions
SRCS=(
    "ru-ar"
    "ar-ru"
    "zh-ar"
    "ar-zh"
    "zh-ru"
    "ru-zh"
)

# We store the results into the text file in ./tmo
echo $MODEL_PATH >> tmp/gen.$MODEL.out

echo "process test data..."
for SRC in "${SRCS[@]}"; do
    echo $SRC >> tmp/gen.$MODEL.out
    array=(${SRC//-/ })
    src=${array[0]}
    tgt=${array[1]}
    echo $src
    echo $tgt

    python data/subword-nmt/subword_nmt/apply_bpe.py -c $TEXT/code < $TEXT/testset/multiun.tokenized.testset.$src > $TEST/test.bpe.$SRC.$src
    python data/subword-nmt/subword_nmt/apply_bpe.py -c $TEXT/code < $TEXT/testset/multiun.tokenized.testset.$tgt > $TEST/test.bpe.$SRC.$tgt
    awk '{print "<2'$tgt'> "$0}' $TMP/test.bpe.$src > $TEST/test.$SRC.$src
    cp $TMP/test.bpe.$tgt $TEST/test.$SRC.$tgt
    
    fairseq-preprocess --source-lang $src --target-lang $tgt \
        --testpref $TEST/test.$SRC \
        --srcdict $DICT --tgtdict $DICT \
        --destdir data-bin/${MODEL}_tests/test_$SRC \
        --workers 20
    
    fairseq-generate data-bin/${MODEL}_tests/test_$SRC/ --path $MODEL_PATH --beam 5 \
        --remove-bpe  --batch-size 128  | tee $TEST/gen.${src}-${tgt}.out
    
    grep ^D $TEST/gen.${src}-${tgt}.out | sort -n -k 2 -t - | cut -f3- | perl $LC | perl $DETOK -l $tgt  > $TEST/gen.${src}-${tgt}.detok.sys

    cat data/MultiUN/testsets/testset/UNv1.0.testset.$tgt | perl $LC > $TEST/gen.${src}-${tgt}.detok.ref

    if [ $tgt = "zh" ] ; then 
        echo "using character-level evaluation"
        cat $TEST/gen.${src}-${tgt}.detok.sys | sacrebleu $TEST/gen.${src}-${tgt}.detok.ref --tokenize zh >> tmp/gen.$MODEL.out
    else 
        cat $TEST/gen.${src}-${tgt}.detok.sys | sacrebleu $TEST/gen.${src}-${tgt}.detok.ref >> tmp/gen.$MODEL.out
    fi
    
done

rm -rf data-bin/${MODEL}_tests
rm -rf $TEST