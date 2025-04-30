import os
import shutil

# Model and data paths configuration
MODEL_CONFIG = {
    'base_model_path': './checkpoints/Qwen2-7B-Instruct',
    'reason_model_output_dir': './checkpoints/reason/',
    'align_model_output_dir': './checkpoints/align/'
}

# Data paths configuration
DATA_CONFIG = {
    'transport_data_path': './ST_data/transport/',
    'align_data_path': './ST_data/align/',
}

# Output paths configuration
OUTPUT_CONFIG = {
    'output_path_1': './outputs1',
    'output_path_2': './outputs2',
    'output_path_3': './outputs3',
    'test_output_path': './test_outputs'
}

# Training configuration
TRAIN_CONFIG = {
    'num_gpus': 2,
    'train_batch_size': 2,
    'eval_batch_size': 16,
    'num_training_iterations': 5
}

def clean_directories():
    """Clean up existing directories and files before training."""
    for dir_path in [MODEL_CONFIG['reason_model_output_dir'], 
                    MODEL_CONFIG['align_model_output_dir']]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    

def clean_output_directories():
    """Clean up output directories."""
    for output_path in [OUTPUT_CONFIG['output_path_1'],
                       OUTPUT_CONFIG['output_path_2'],
                       OUTPUT_CONFIG['output_path_3']]:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

def run_training_scripts():
    """Execute the training and data generation scripts."""

    # Run pretraining script
    os.system('bash stq_pretrain.sh {} {} {} {} {} {}'.format(
        MODEL_CONFIG['base_model_path'],
        DATA_CONFIG['align_data_path'],
        MODEL_CONFIG['align_model_output_dir'],
        TRAIN_CONFIG['num_gpus'],
        TRAIN_CONFIG['train_batch_size'],
        TRAIN_CONFIG['eval_batch_size']
    ))
        
    # Run main training loop
    for i in range(TRAIN_CONFIG['num_training_iterations']):
        # Run training script
        os.system('bash stq_train.sh {} {} {} {} {} {}'.format(
            MODEL_CONFIG['base_model_path'],
            DATA_CONFIG['transport_data_path'],
            MODEL_CONFIG['reason_model_output_dir'],
            TRAIN_CONFIG['num_gpus'],
            TRAIN_CONFIG['train_batch_size'],
            TRAIN_CONFIG['eval_batch_size']
        ))

        # Clean output directories
        clean_output_directories()
        
        # Generate data for training
        for output_path in [OUTPUT_CONFIG['output_path_1'],
                            OUTPUT_CONFIG['output_path_2'],
                            OUTPUT_CONFIG['output_path_3']]:
            os.system('bash stq_make_data.sh {} {} {} {} {} {} {}'.format(
                MODEL_CONFIG['reason_model_output_dir'],
                MODEL_CONFIG['base_model_path'],
                DATA_CONFIG['transport_data_path'],
                output_path,
                TRAIN_CONFIG['num_gpus'],
                TRAIN_CONFIG['eval_batch_size'],
                'train'
                ))

        if i == 0:
            # Generate test data
            os.system('bash stq_make_data.sh {} {} {} {} {} {} {}'.format(
                MODEL_CONFIG['reason_model_output_dir'],
                MODEL_CONFIG['base_model_path'],
                DATA_CONFIG['transport_data_path'],
                OUTPUT_CONFIG['test_output_path'],
                TRAIN_CONFIG['num_gpus'],
                TRAIN_CONFIG['eval_batch_size'],
                'test'
            ))

        else:
            # Generate test data
            os.system('bash stq_eval.sh {} {} {} {} {} {} {}'.format(
                MODEL_CONFIG['reason_model_output_dir'],
                MODEL_CONFIG['base_model_path'],
                DATA_CONFIG['transport_data_path'],
                OUTPUT_CONFIG['test_output_path'],
                TRAIN_CONFIG['num_gpus'],
                TRAIN_CONFIG['eval_batch_size'],
                'test'
            ))

    


if __name__ == "__main__":

    # Run training script
    os.system('bash stq_train.sh {} {} {} {} {} {}'.format(
        MODEL_CONFIG['base_model_path'],
        DATA_CONFIG['transport_data_path'],
        MODEL_CONFIG['reason_model_output_dir'],
        TRAIN_CONFIG['num_gpus'],
        TRAIN_CONFIG['train_batch_size'],
        TRAIN_CONFIG['eval_batch_size']
    ))

    # clean_directories()
    # run_training_scripts()
    