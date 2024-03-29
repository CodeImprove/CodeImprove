python ./rundissector.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=roberta-base \
    --model_name_or_path=roberta-base \
    --dissector \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/dev.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee test.log

#python ../evaluator/evaluator.py -a ../dataset/test.jsonl -p saved_models/predictions.txt
