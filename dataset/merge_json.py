import json
import os

# file and dataset paths

file_name = 'train.json' # the EMGF dataset file containing the dep and con spans
amrbart_file_name = 'train_with_amr.json' # the AMRBART dataset file containing the amr edges and tokens
dataset_name = 'Laptops_spring' # the name of the dataset folder containing the files
output_file_name = 'train_spring.json' # the name of the output file to be created

# Define file paths
current_dir = os.path.dirname(os.path.abspath(__file__)) 
current_dir = os.path.join(current_dir, dataset_name)

test_new_path = os.path.join(current_dir, file_name)
test_with_amrbart_path = os.path.join(current_dir, amrbart_file_name)
output_path = os.path.join(current_dir, output_file_name)

# Read test_new.json
with open(test_new_path, 'r') as file:
    test_new_data = json.load(file)

# Read test_with_amrbart.json
with open(test_with_amrbart_path, 'r') as file:
    amrbart_data = json.load(file)

# Prepare merged output
merged_output = []

# Merge data from both files
for i in range(len(test_new_data)):
    new_item = test_new_data[i]
    amr_item = amrbart_data[i]
    
    # Create merged item with specified keys
    merged_item = {
        'token': new_item.get('token', []),
        'pos': new_item.get('pos', []),
        'head': new_item.get('head', []),
        'deprel': new_item.get('deprel', []),
        'aspects': new_item.get('aspects', []),
        'dep_head': new_item.get('dep_head', []),
        'con_head': new_item.get('con_head', []),
        'con_mapnode': new_item.get('con_mapnode', []),
        'aa_choice': new_item.get('aa_choice', []),
        'amr_edges': amr_item.get('amr_edges', []),
        'amr_tokens': amr_item.get('amr_tokens', [])
    }
    if amr_item.get('amr_tokens', []) != new_item.get('token', []):
        print(f"Warning: AMR tokens do not match input tokens for index {i}", new_item.get('token', []), amr_item.get('amr_tokens', []))
    merged_output.append(merged_item)

# Write to merged_output.json
with open(output_path, 'w') as file:
    json.dump(merged_output, file, indent=4)

print(f"Successfully merged data and saved to {output_path}")