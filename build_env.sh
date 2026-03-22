export MAX_JOBS=4

# conda create --prefix /home/loujieming.ljm/.conda/envs/sdpo_env python=3.10

source /opt/conda/bin/activate
conda activate sdpo_env
export PATH="/home/loujieming.ljm/.conda/envs/sdpo_env/bin:$PATH"
# export PATH="/opt/conda/envs/python3.10/bin:$PATH"
# conda activate python3.10

bash experiments/generalization/run_baseline_grpo_all_local.sh |tee logs/run_baseline_grpo_all_local_$(date +%Y-%m-%d_%H-%M-%S).log
bash experiments/generalization/run_sdpo_all_local.sh |tee logs/run_sdpo_all_local_$(date +%Y-%m-%d_%H-%M-%S).log

CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/rich_feedback/run_baseline_grpo_local.sh |tee logs/run_baseline_grpo_rich_feedback_local_$(date +%Y-%m-%d_%H-%M-%S).log
CUDA_VISIBLE_DEVICES=4,5,6,7 bash experiments/rich_feedback/run_sdpo_local.sh |tee logs/run_sdpo_rich_feedback_local_$(date +%Y-%m-%d_%H-%M-%S).log

CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/generalization/run_sdpo_all_entropy_weighting_local_1.sh |tee logs/run_sdpo_all_entropy_weighting_local_1_$(date +%Y-%m-%d_%H-%M-%S).log
CUDA_VISIBLE_DEVICES=4,5,6,7 bash experiments/generalization/run_sdpo_all_entropy_weighting_local_2.sh |tee logs/run_sdpo_all_entropy_weighting_local_2_$(date +%Y-%m-%d_%H-%M-%S).log

CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/generalization/run_tasd_all_local_1.sh |tee logs/run_tasd_all_local_1_$(date +%Y-%m-%d_%H-%M-%S).log
CUDA_VISIBLE_DEVICES=4,5,6,7 bash experiments/generalization/run_tasd_all_local_2.sh |tee logs/run_tasd_all_local_2_$(date +%Y-%m-%d_%H-%M-%S).log

CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/generalization/run_tasd_relative_rewards_local.sh |tee logs/run_tasd_relative_rewards_local_$(date +%Y-%m-%d_%H-%M-%S).log

CUDA_VISIBLE_DEVICES=4,5,6,7 bash experiments/generalization/run_tasd_teacher_prob_local.sh |tee logs/run_tasd_teacher_prob_local_$(date +%Y-%m-%d_%H-%M-%S).log

swanlab watch -l /home/loujieming.ljm/swanlab_logs

pip install transformer-engine[pytorch] --trusted-host mirrors.aliyun.com -i http://mirrors.aliyun.com/pypi/simple/

# pip cache purge

pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu128 --trusted-host mirrors.aliyun.com -i http://mirrors.aliyun.com/pypi/simple/

# pip install -r requirements.txt --trusted-host mirrors.aliyun.com -i http://mirrors.aliyun.com/pypi/simple/

pip install -r requirements_flex.txt --trusted-host mirrors.aliyun.com -i http://mirrors.aliyun.com/pypi/simple/

# Install SDPO (verl) in editable mode
pip install -e . --trusted-host mirrors.aliyun.com -i http://mirrors.aliyun.com/pypi/simple/
 
# Install Flash Attention 2 (compiled from source)
export MAX_JOBS=4
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-build-isolation --trusted-host mirrors.aliyun.com -i http://mirrors.aliyun.com/pypi/simple/

# pip install -r requirements_sglang.txt --trusted-host mirrors.aliyun.com -i http://mirrors.aliyun.com/pypi/simple/