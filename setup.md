# Step 1: Activate Virtual Environement
In the root directory of the code base "federated_librispeech", type `source flvenv/bin/activate`

# Step 2: Load neccessary modules
`module load StdEnv/2023`
`module load scipy-stack/2025a`

# Step 3: Validate the configuration details.
Ensure that the dataset paths, split and model configuration are correct.

# Step 4: Run the partitioning sceme.
Allocate necessary resources for it, it may take a while.
Run the `partition_data.py` file.

# Additional Claude Setup
source flvenv/bin/activate
module load StdEnv/2023
module load nodejs/20.16.0
./node_modules/.bin/claude

# To allocated resources
salloc --account=def-aravila --time=8:00:00 --mem=64G --cpus-per-task=4 --gres=gpu:2

# For TMUX Sessions 
1. tmux new-session -s name
2. tmux deattach -t name
3. tmux attach -t name
4. tmux ls
5. exit

# Finetuning and s3prl Instructions:

1.Basic Syntax: python3 run_downstream.py -m train -u fbank -d speech_commands -n ExpName

2.Evaluation: python3 run_downstream.py -m evaluate -e result/downstream/ExpName/dev-best.ckpt

With custom upstream model: python3 /home/saadan/scratch/federated_librispeech/s3prl/s3prl/run_downstream.py -m train -u  hubert_local -k  /home/saadan/scratch/federated_librispeech/src/checkpoints/pretraining/server/best_global_model.pt -d  speech_commands -c /home/saadan/scratch/federated_librispeech/src/configs/speech_commands_config.yaml -p   /home/saadan/scratch/federated_librispeech/src/exp/speech_commands -s last_hidden_state

The key parameters are:
- -m: mode (train/evaluate)
- -u: upstream model (fbank, wav2vec2, hubert, etc.)
- -d: downstream task (speech_commands)
- -n: experiment name
- -k: checkpoint path (optional)
- -e: evaluation checkpoint path