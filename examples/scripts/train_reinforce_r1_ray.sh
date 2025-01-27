# reinforce++
export DATASET="/root/projects/OpenRLHF/data/mathlv345_9k.json"
export MODEL="DeepSeek-R1-Distill-Qwen-1.5B"
python -m openrlhf.models.remote_rm.math_verifier --dataset $DATASET > remote_rm.log 2>&1 &
childpid=$!

ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/root/projects/OpenRLHF"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --remote_rm_url http://127.0.0.1:5000//get_reward \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --pretrain /root/projects/OpenRLHF/ckpts/${MODEL} \
   --save_path /root/projects/OpenRLHF/ckpts/${MODEL}-reinforce \
   --micro_train_batch_size 4 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 1024 \
   --temperature 0.6 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 3000 \
   --advantage_estimator reinforce \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --init_kl_coef 0.01 \
   --prompt_data $DATASET \
   --input_key prompt \
   --normalize_reward \
   --flash_attn \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps 4 \
   --ckpt_path /root/projects/OpenRLHF/ckpts/${MODEL}-reinforce_ckpts \
   --use_wandb 9aeddea3b60542704fd5cd44d4c4a1d1d911ce54 \
   --wandb_run_name ${MODEL}-reinforce

# also supports --advantage_estimator rloo
ray stop
kill $childpid