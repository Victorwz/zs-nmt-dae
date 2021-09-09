fairseq-preprocess --source-lang src --target-lang tgt \
    --trainpref data/MultiUN-raw-data-2M/mnmt-dn/train.clean \
    --validpref data/MultiUN-raw-data-2M/mnmt-dn/valid \
    --destdir data-bin/multiun_mnmt_denoising/ \
    --workers 20 --joined-dictionary

# The output dictionary will be saved in data-bin/multiun_mnmt_denoising/dict.src.txt
# The dictionary will be used for decoding and testing

# We use 4 Nvidia 16GB-memory P100 GPUs for training
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train \
     data-bin/multiun_mnmt_denoising \
    --task translation \
    --arch transformer_wmt_en_de --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.1 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --scoring sacrebleu \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --max-update 500000 --save-interval-updates 10000  \
    --save-dir checkpoints/multiun_mnmt_denoising/