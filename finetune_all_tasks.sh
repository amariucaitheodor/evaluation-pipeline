#!/bin/bash

MODEL_PATH=$1
LR=${2:-2e-5}
PATIENCE=${3:-10}
BSZ=${4:-32}
EVAL_EVERY=${5:-200}
MAX_EPOCHS=${6:-10}
SEED=${7:-12}

# If your system uses sbatch or qsub, consider using that to parallelize calls to finetune_model.sh

# We use custom values from https://github.com/facebookresearch/fairseq/tree/main/examples/roberta/config/finetuning
# GLUE tasks
sbatch ./finetune_model.sh $MODEL_PATH glue "cola" 1e-5 $PATIENCE 16 $EVAL_EVERY $MAX_EPOCHS $SEED 5336 # 5110 auto
sbatch ./finetune_model.sh $MODEL_PATH glue "mnli" 1e-5 $PATIENCE 32 $EVAL_EVERY $MAX_EPOCHS $SEED 123873 # 81190 auto
sbatch ./finetune_model.sh $MODEL_PATH glue "mrpc" 1e-5 $PATIENCE 16 $EVAL_EVERY $MAX_EPOCHS $SEED 2296
sbatch ./finetune_model.sh $MODEL_PATH glue "qnli" 1e-5 $PATIENCE 32 $EVAL_EVERY $MAX_EPOCHS $SEED 33112
sbatch ./finetune_model.sh $MODEL_PATH glue "qqp" 1e-5 $PATIENCE 32 $EVAL_EVERY $MAX_EPOCHS $SEED 113272
sbatch ./finetune_model.sh $MODEL_PATH glue "rte" 2e-5 $PATIENCE 16 $EVAL_EVERY $MAX_EPOCHS $SEED 2036
sbatch ./finetune_model.sh $MODEL_PATH glue "sst2" 1e-5 $PATIENCE 32 $EVAL_EVERY $MAX_EPOCHS $SEED 20935
sbatch ./finetune_model.sh $MODEL_PATH glue "mnli-mm" $LR $PATIENCE $BSZ $EVAL_EVERY $MAX_EPOCHS $SEED # multiNLI mismatched

# SuperGLUE tasks
sbatch ./finetune_model.sh $MODEL_PATH glue "multirc" $LR $PATIENCE $BSZ $EVAL_EVERY $MAX_EPOCHS $SEED
sbatch ./finetune_model.sh $MODEL_PATH glue "wsc" $LR $PATIENCE $BSZ $EVAL_EVERY $MAX_EPOCHS $SEED
sbatch ./finetune_model.sh $MODEL_PATH glue "boolq" $LR $PATIENCE $BSZ $EVAL_EVERY $MAX_EPOCHS $SEED
# ./finetune_model.sh theodor1289/thesis_halfsize_text1_vision0 glue cola 1e-5 10 16 200 10 12 5336

# Fine-tune and evaluate on MSGS tasks
#for subtask in {"main_verb_control","control_raising_control","syntactic_category_control","lexical_content_the_control","relative_position_control","main_verb_lexical_content_the","main_verb_relative_token_position","syntactic_category_lexical_content_the","syntactic_category_relative_position","control_raising_lexical_content_the","control_raising_relative_token_position"}; do
#	./finetune_model.sh $MODEL_PATH msgs $subtask $LR $PATIENCE $BSZ $EVAL_EVERY $MAX_EPOCHS $SEED
#done
