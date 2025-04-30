# to fill in the following path to evaluation!
model_name=$1
base_model=$2
data_path=$3
output_res_path=$4
num_gpus=$5
num_per_gpu=$6
data_tag=$7

python ./tsgpt/eval/run_data.py --num_gpus ${num_gpus} --batch_size ${num_per_gpu} --data_tag ${data_tag} --model-name ${model_name} --base-model ${base_model} --prompting_file ${data_path} --st_data_path ${data_path} --output_res_path ${output_res_path}
