#!/bin/bash

MODEL_PATH=$1
LR=${2:-2e-5}
PATIENCE=${3:-10}
BSZ=${4:-32}
EVAL_EVERY=${5:-200}
MAX_EPOCHS=${6:-10}
SEED=${7:-12}

# We use custom values from https://github.com/facebookresearch/fairseq/tree/main/examples/roberta/config/finetuning
# GLUE tasks
sbatch --job-name="finetune" --time=1-00:00:00 --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus=1 --cpus-per-task=4 \
  --mem-per-cpu=10000 --gres=gpumem:20g --output "finetune_$MODEL_PATH/cola.log" --wrap="./finetune_model.sh $MODEL_PATH glue cola 1e-5 $PATIENCE 16 $EVAL_EVERY $MAX_EPOCHS $SEED 5336" # 5110 auto
sbatch --job-name="finetune" --time=1-00:00:00 --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus=1 --cpus-per-task=4 \
  --mem-per-cpu=10000 --gres=gpumem:20g --output "finetune_$MODEL_PATH/mnli.log" --wrap="./finetune_model.sh $MODEL_PATH glue mnli 1e-5 $PATIENCE 32 $EVAL_EVERY $MAX_EPOCHS $SEED 123873" # 81190 auto
sbatch --job-name="finetune" --time=1-00:00:00 --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus=1 --cpus-per-task=4 \
  --mem-per-cpu=10000 --gres=gpumem:20g --output "finetune_$MODEL_PATH/mrpc.log" --wrap="./finetune_model.sh $MODEL_PATH glue mrpc 1e-5 $PATIENCE 16 $EVAL_EVERY $MAX_EPOCHS $SEED 2296"
sbatch --job-name="finetune" --time=1-00:00:00 --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus=1 --cpus-per-task=4 \
  --mem-per-cpu=10000 --gres=gpumem:20g --output "finetune_$MODEL_PATH/qnli.log" --wrap="./finetune_model.sh $MODEL_PATH glue qnli 1e-5 $PATIENCE 32 $EVAL_EVERY $MAX_EPOCHS $SEED 33112"
sbatch --job-name="finetune" --time=1-00:00:00 --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus=1 --cpus-per-task=4 \
  --mem-per-cpu=10000 --gres=gpumem:20g --output "finetune_$MODEL_PATH/qqp.log" --wrap="./finetune_model.sh $MODEL_PATH glue qqp 1e-5 $PATIENCE 32 $EVAL_EVERY $MAX_EPOCHS $SEED 113272"
sbatch --job-name="finetune" --time=1-00:00:00 --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus=1 --cpus-per-task=4 \
  --mem-per-cpu=10000 --gres=gpumem:20g --output "finetune_$MODEL_PATH/rte.log" --wrap="./finetune_model.sh $MODEL_PATH glue rte 2e-5 $PATIENCE 16 $EVAL_EVERY $MAX_EPOCHS $SEED 2036"
sbatch --job-name="finetune" --time=1-00:00:00 --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus=1 --cpus-per-task=4 \
  --mem-per-cpu=10000 --gres=gpumem:20g --output "finetune_$MODEL_PATH/sst2.log" --wrap="./finetune_model.sh $MODEL_PATH glue sst2 1e-5 $PATIENCE 32 $EVAL_EVERY $MAX_EPOCHS $SEED 20935"
sbatch --job-name="finetune" --time=1-00:00:00 --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus=1 --cpus-per-task=4 \
  --mem-per-cpu=10000 --gres=gpumem:20g --output "finetune_$MODEL_PATH/mnli-mm.log" --wrap="./finetune_model.sh $MODEL_PATH glue mnli-mm $LR $PATIENCE $BSZ $EVAL_EVERY $MAX_EPOCHS $SEED" # multiNLI mismatched

# SuperGLUE tasks
sbatch --job-name="finetune" --time=1-00:00:00 --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus=1 --cpus-per-task=4 \
  --mem-per-cpu=10000 --gres=gpumem:20g --output "finetune_$MODEL_PATH/multirc.log" --wrap="./finetune_model.sh $MODEL_PATH glue multirc $LR $PATIENCE $BSZ $EVAL_EVERY $MAX_EPOCHS $SEED"
sbatch --job-name="finetune" --time=1-00:00:00 --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus=1 --cpus-per-task=4 \
  --mem-per-cpu=10000 --gres=gpumem:20g --output "finetune_$MODEL_PATH/wsc.log" --wrap="./finetune_model.sh $MODEL_PATH glue wsc $LR $PATIENCE $BSZ $EVAL_EVERY $MAX_EPOCHS $SEED"
sbatch --job-name="finetune" --time=1-00:00:00 --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus=1 --cpus-per-task=4 \
  --mem-per-cpu=10000 --gres=gpumem:20g --output "finetune_$MODEL_PATH/boolq.log" --wrap="./finetune_model.sh $MODEL_PATH glue boolq $LR $PATIENCE $BSZ $EVAL_EVERY $MAX_EPOCHS $SEED"
# ./finetune_model.sh theodor1289/thesis_halfsize_text1_vision0 glue cola 1e-5 10 16 200 10 12 5336
