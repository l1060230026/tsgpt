'''读取outputs1里面的所有json文件'''
import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process data files')
    parser.add_argument('--data_path', type=str, default='ST_data/transport', help='Path to save the output data')
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path

    self_train = []

    outputs1_dir = 'outputs1'
    outputs2_dir = 'outputs2'
    outputs3_dir = 'outputs3'
    generate_data = []

    json_files = [f for f in os.listdir(outputs1_dir) if f.endswith('.json')]

    for file in json_files:
        with open(os.path.join(outputs1_dir, file), 'r') as f:
            results = json.load(f)
            generate_data.extend(results)

    '''读取results.json文件'''
    import json

    with open('results.json', 'r') as f:
        results = json.load(f)

    score_indices = results['score_indices']
    mean_score = results['current_average']
    max_score = 10
    metric = (max_score + mean_score) / 2

    for i in range(len(generate_data)):
        assert generate_data[i]['id'] == score_indices[i]['id']

        if score_indices[i]['score'] > metric:
            info = {
                'question': generate_data[i]['question'],
                'role': generate_data[i]['role'],
                'timeseries_operation': generate_data[i]['timeseries_operation'],
                'answer': generate_data[i]['prediction'],
                'rationales': generate_data[i]['rationales'],
                'df_id': generate_data[i]['df_id']
            }
            self_train.append(info)
        else:
            continue


    # 读取outputs1文件夹下的所有json文件
    files = os.listdir(outputs1_dir)
    outputs1_data = []
    for file in files:
        with open(os.path.join(outputs1_dir, file), 'r') as f:
            items = json.load(f)
            for item in items:
                outputs1_data.append(item)

    # 读取outputs2文件夹下的所有json文件
    files = os.listdir(outputs2_dir)
    outputs2_data = []
    for file in files:
        with open(os.path.join(outputs2_dir, file), 'r') as f:
            items = json.load(f)
            for item in items:
                outputs2_data.append(item)

    # 读取outputs3文件夹下的所有json文件
    files = os.listdir(outputs3_dir)
    outputs3_data = []
    for file in files:
        with open(os.path.join(outputs3_dir, file), 'r') as f:
            items = json.load(f)
            for item in items:
                outputs3_data.append(item)

    for i in range(len(outputs1_data)):
        assert outputs1_data[i]['question'] == outputs2_data[i]['question'] == outputs3_data[i]['question']
        
        rationales = outputs1_data[i]['rationales']
        solutions = 'Solution1:\nReasoning:\n'
        for rationale in rationales:
            solutions += f'{rationale}\n'
        solutions += f'Answer: {outputs1_data[i]["prediction"]}'
        rationales = outputs2_data[i]['rationales']
        solutions += '\nSolution2:\nReasoning:\n'
        for rationale in rationales:
            solutions += f'{rationale}\n'
        solutions += f'Answer: {outputs1_data[i]["prediction"]}'
        rationales = outputs3_data[i]['rationales']
        solutions += '\nSolution3:\nReasoning:\n'
        for rationale in rationales:
            solutions += f'{rationale}\n'
        solutions += f'Answer: {outputs3_data[i]["prediction"]}'

        info = {
            'question': outputs1_data[i]['question'],
            'role': outputs1_data[i]['role'],
            'timeseries_operation': outputs1_data[i]['timeseries_operation'],
            'df_id': outputs1_data[i]['df_id'],
            'solutions': solutions,
            'answer': outputs1_data[i]['groundtruth'],
        }
        

        generate_data.append(info)

    with open(f'{data_path}/self_train.jsonl', 'w') as f:
        for item in generate_data:
            json.dump(item, f)
            f.write('\n')

if __name__ == '__main__':
    main()

