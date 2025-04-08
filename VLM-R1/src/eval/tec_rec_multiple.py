from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
from pprint import pprint
import random
import json
import concurrent.futures

BSZ=8
DATA_ROOT = "/home/ec2-user/VLM-R1/src/data/datasets/"

TEST_DATASETS = ['vlm_r1_test']
IMAGE_ROOT = "/home/ec2-user/VLM-R1/src/data/datasets/vlm_r1_test"


def extract_count_answer(content):
    # Try to find the count within <answer> tags, if can not find, return -1
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    count_pattern = r'(\d+)'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        count_match = re.search(count_pattern, content_answer)
        if count_match:
            try:
                count = int(count_match.group(1))
                return count
            except:
                return None
    return None


def run_datasets(processor, model, output_path, gpu_id):
    for ds in TEST_DATASETS:
        print(f"Processing {ds}...")
        ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
        data = json.load(open(ds_path, "r"))
        random.shuffle(data)
        QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
        messages = []

        for x in data:
            image_path = os.path.join(IMAGE_ROOT, x['image'])
            message = [
                # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": f"file://{image_path}"
                    },
                    {
                        "type": "text",
                        "text": QUESTION_TEMPLATE.format(Question=x['problem'])
                    }
                ]
            }]
            messages.append(message)

        all_outputs = []  # List to store all answers

        # Process data
        for i in tqdm(range(0, len(messages), BSZ), 'Run for {output_path}'):
            batch_messages = messages[i:i + BSZ]
        
            # Preparation for inference
            text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
            
            image_inputs, video_inputs = process_vision_info(batch_messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(f"cuda:{gpu_id}")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            all_outputs.extend(batch_output_text)

        final_output = []
        correct_number = 0
        no_answer = 0

        for input_example, model_output in zip(data, all_outputs):
            original_output = model_output
            ground_truth = input_example['solution']
            
            model_answer = extract_count_answer(original_output)
            correct = 0
            if model_answer is not None:
                if model_answer == ground_truth:
                    correct = 1
            else:
                no_answer += 1
            
            correct_number += correct
            
            # Create a result dictionary for this example
            result = {
                'original_input': input_example,
                'question': input_example['problem'],
                'ground_truth': ground_truth,
                'model_output': original_output,
                'extracted_answer': model_answer,
                'correct': correct
            }
            final_output.append(result)

        # Calculate and print accuracy
        accuracy = correct_number / len(data) * 100
        print(f"\nAccuracy of {ds}: {accuracy:.2f}%")

        # Save results to a JSON file
        with open(output_path, "w") as f:
            json.dump({
                'accuracy': accuracy,
                'no_answer': no_answer,
                'results': final_output
            }, f, indent=2)

        print(f"Results saved to {output_path}")
        print("-"*100)


def get_models():
    model_folders = json.load(open("model_folders.json", "r"))
    models = []
    
    for folder in model_folders:
        for steps in range(1, 1000):
            model_path = f"/home/ec2-user/VLM-R1/src/open-r1-multimodal/output/{folder}/checkpoint-{steps}"
            if os.path.exists(model_path):
                models.append({
                    'model_path': model_path,
                    'folder': folder,
                    'steps': steps,
                    'output_path': f"./logs/rec_results_{folder}_{steps}.json"
                })
    
    return models

# Function to process a single model on a specific GPU
def process_model(model_info, gpu_id):
    print(f"Running model {model_info['folder']} at step {model_info['steps']} on GPU {gpu_id}...")

    model_path = model_info['model_path']
    output_path = model_info['output_path']
    
    # Assign the specific GPU
    device = f"cuda:{gpu_id}"
    
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,  # Assign the model to the designated GPU
    )
    
    run_datasets(processor, model, output_path, gpu_id)
    
    print(f"Model {model_info['folder']} at step {model_info['steps']} completed on GPU {gpu_id}.")
    print("-" * 100)


def run_parallel():
    models = get_models()
    print("Models found:", len(models))

    num_gpus = torch.cuda.device_count()  # Get available GPUs
    print(f"Using {num_gpus} GPUs.")

    for model_info in tqdm(models, 'Running models'):
        process_model(model_info, 0)
  
if __name__ == "__main__":
    run_parallel()