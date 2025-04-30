import json
import os
from pathlib import Path
import time
from openai import OpenAI
import numpy as np
from multiprocessing import Pool, Value, Lock
from tqdm import tqdm
import random

system_prompt = '''You are a helpful and precise assistant for evaluating the quality of AI-generated answers against a reference answer.

We would like to request your feedback on the performance of an AI assistant's response compared to a reference answer for a given question about time series data.

The time series data typically includes:
- A sequence of numerical values over time
- Timestamps for each data point
- Relevant metadata (e.g., sampling frequency, units, etc.)

Please rate the following aspects of their responses:
- Helpfulness: How well the answer addresses the user's question
- Relevance: How directly the response relates to the time series data
- Accuracy: How accurate the analysis and interpretation of the data is
- Level of details: How thorough and detailed the explanation is

Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.

IMPORTANT: You MUST follow this exact output format:

Score: [number between 1-10]

Evaluation:
[Your detailed evaluation here]

[Question] {question}
[Reference Answer] {answer}
[End of Reference Answer]
[AI Assistant's Response] {answer_2}
[End of AI Assistant's Response]'''


api_keys = ['78a829522304f98d99428126394a2bf4',
            'd857a1c395da16b7e96221323f1dafd5', 
            '08debd5db005aea535b052a4442e1163',
            # 'e2f0b576d86dcc538ae101fc02150dc3',
            'becd3719f15ec4a8482eefc805559a44',
            '9a2a0faf11f6ae90e8eb165f789bd8b1',
            '92baa264e510d1b89ed7f8adf793a87c',
            '78a5582c9e6c5c6d562fb2211e5c09d7'
            ]
base_urls = "https://idealab.alibaba-inc.com/api/openai/v1"


# model_id = "qwen2.5-72b-instruct"
model_id = "qwen2.5-max"
# model_id = "qwen-max-2025-01-25-chat"

class LoadBalancer:
    def __init__(self, num_api_keys):
        self.num_api_keys = num_api_keys
        self.counters = [Value('i', 0) for _ in range(num_api_keys)]
        self.locks = [Lock() for _ in range(num_api_keys)]
    
    def get_next_api_key(self):
        # Find the API key with minimum usage
        min_count = float('inf')
        min_index = 0
        
        for i in range(self.num_api_keys):
            with self.locks[i]:
                if self.counters[i].value < min_count:
                    min_count = self.counters[i].value
                    min_index = i
        
        # Increment the counter for the selected API key
        with self.locks[min_index]:
            self.counters[min_index].value += 1
        
        return min_index

# Create a global load balancer instance
load_balancer = LoadBalancer(len(api_keys))

def get_client(process_id):
    return OpenAI(
        api_key=api_keys[process_id],
        base_url=base_urls,
    )

def llm_request(text, process_id):
    client = get_client(process_id)
    timeout = 30  # seconds
    
    while True:
        try:
            chat_response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a QAR (Question-Answer-Rationale) validator."},
                    {"role": "user", "content": [{"type": "text", "text": text}]},
                ],
                timeout=timeout
            )
            response = chat_response
            message_content = response.choices[0].message.content
            if message_content == "Request error occurred: ":
                time.sleep(random.randint(1, 5))  # 随机休息1-5s
                continue
            else:
                return message_content
        except Exception as e:
            # print(f"Error: {e}")
            time.sleep(random.randint(1, 5))  #随机休息1-5s
            continue

def extract_score(text):
    """Extract the score from a string that starts with 'Score: '"""
    try:
        # Find the line that starts with 'Score: '
        score_line = next(line for line in text.split('\n') if line.startswith('Score: '))
        # Extract the number after 'Score: '
        score = int(score_line.split('Score: ')[1].strip())
        return score
    except (StopIteration, ValueError, IndexError):
        return None

def process_data(data):
    try:
        # Get the next available API key index
        api_key_index = load_balancer.get_next_api_key()
        question = data["question"]
        answer = data["groundtruth"]
        answer_2 = data["prediction"]
        score = llm_request(system_prompt.format(question=question, answer=answer, answer_2=answer_2), api_key_index)
        score = extract_score(score)
        data['score'] = score
        return data
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        data['score'] = 0
        return data

if __name__ == "__main__":
    outputs_dir = Path("outputs")
    results_file = Path("results.json")
    
    # Get all JSON files in the directory
    json_files = list(outputs_dir.glob("*.json"))
    
    # Dictionary to store all data
    all_data = []
    
    # Read each JSON file
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.extend(data)
    
    # Load previous results if exists
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            previous_results = [json.loads(line) for line in f]
            processed_count = previous_results[-1].get('processed_count', 0) if previous_results else 0
            score_indices = [item for item in previous_results if 'score' in item]
    else:
        processed_count = 0
        score_indices = []
    
    # Skip already processed items
    remaining_data = all_data[processed_count:]

    num_processes = 16 * len(api_keys)
    
    # Use multiprocessing with 16 processes
    with Pool(num_processes) as pool:
        for i, item in enumerate(tqdm(pool.imap(process_data, remaining_data), 
                                     total=len(remaining_data), 
                                     desc="Processing")):
            score_indices.append(item)
            processed_count += 1
            
            # Save results every 100 items or at the end
            if (i + 1) % 100 == 0 or i == len(remaining_data) - 1:
                valid_scores = [s["score"] for s in score_indices if s["score"] > 0]
                current_stats = {
                    'processed_count': processed_count,
                    'score_indices': score_indices,
                    'current_average': np.mean(valid_scores) if valid_scores else 0,
                    'success_count': len(valid_scores),
                    'failure_count': len(score_indices) - len(valid_scores)
                }
                
                # Save score_indices in JSON
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(current_stats, f, indent=2)
                
                print(f"\nProgress Update:")
                print(f"Total processed: {processed_count}")
                print(f"Successfully processed: {len(valid_scores)}")
                print(f"Failed: {len(score_indices) - len(valid_scores)}")
                if valid_scores:
                    print(f"Current average score: {np.mean(valid_scores):.2f}")
    
    # Final statistics
    valid_scores = [s["score"] for s in score_indices if s["score"] > 0]
    print(f"\nFinal Results:")
    print(f"Total processed: {len(score_indices)}")
    print(f"Successfully processed: {len(valid_scores)}")
    print(f"Failed: {len(score_indices) - len(valid_scores)}")
    if valid_scores:
        print(f"Final average score: {np.mean(valid_scores):.2f}")
    else:
        print("No valid scores to calculate average")

