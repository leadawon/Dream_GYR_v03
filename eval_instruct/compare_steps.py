import json
from pathlib import Path

def load_jsonl(filepath):
    """Load JSONL file and return list of dicts"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def compare_files(file_256, file_128, output_file):
    """Compare 256step and 128step files for flexible-extract filter"""
    
    # Load both files
    data_256 = load_jsonl(file_256)
    data_128 = load_jsonl(file_128)
    
    # Create dictionary for easy lookup by doc_id
    dict_256 = {item['doc_id']: item for item in data_256}
    dict_128 = {item['doc_id']: item for item in data_128}
    
    # Find items with flexible-extract filter and different resps
    differences = []
    
    for doc_id in dict_256.keys():
        if doc_id not in dict_128:
            continue
            
        item_256 = dict_256[doc_id]
        item_128 = dict_128[doc_id]
        
        # Check if filter is flexible-extract (need to check filtered_resps field)
        # Looking at the structure, items have 'filter' field
        filter_256 = item_256.get('filter', '')
        filter_128 = item_128.get('filter', '')
        
        # Get resps
        resps_256 = item_256.get('resps', [[]])[0]
        resps_128 = item_128.get('resps', [[]])[0]
        
        # Convert to string if list
        if isinstance(resps_256, list) and len(resps_256) > 0:
            resps_256 = resps_256[0]
        if isinstance(resps_128, list) and len(resps_128) > 0:
            resps_128 = resps_128[0]
            
        # Get filtered_resps
        filtered_resps_256 = item_256.get('filtered_resps', [''])[0] if 'filtered_resps' in item_256 else ''
        filtered_resps_128 = item_128.get('filtered_resps', [''])[0] if 'filtered_resps' in item_128 else ''
        
        # Get exact_match
        exact_match_256 = item_256.get('exact_match', None)
        exact_match_128 = item_128.get('exact_match', None)
        
        # Skip if resps are identical
        if resps_256 == resps_128:
            continue
            
        # Add to differences list
        differences.append({
            'doc_id': doc_id,
            'question': item_256['doc']['question'],
            'target': item_256['target'],
            'filter_256': filter_256,
            'filter_128': filter_128,
            '256_resps': resps_256,
            '128_resps': resps_128,
            '256_filtered_resps': filtered_resps_256,
            '128_filtered_resps': filtered_resps_128,
            '256_exact_match': exact_match_256,
            '128_exact_match': exact_match_128
        })
    
    # Write comparison to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("COMPARISON: 256 STEP vs 128 STEP (Different Resps Only)\n")
        f.write("=" * 100 + "\n\n")
        
        for idx, diff in enumerate(differences, 1):
            f.write(f"\n{'=' * 100}\n")
            f.write(f"CASE #{idx} - Doc ID: {diff['doc_id']}\n")
            f.write(f"{'=' * 100}\n\n")
            
            f.write(f"QUESTION:\n{diff['question']}\n\n")
            f.write(f"TARGET ANSWER: {diff['target']}\n\n")
            
            f.write(f"FILTER (256 step): {diff['filter_256']}\n")
            f.write(f"FILTER (128 step): {diff['filter_128']}\n\n")
            
            f.write("-" * 100 + "\n")
            f.write("256 STEP RESPONSE:\n")
            f.write("-" * 100 + "\n")
            f.write(f"{diff['256_resps']}\n\n")
            
            f.write("-" * 100 + "\n")
            f.write("128 STEP RESPONSE:\n")
            f.write("-" * 100 + "\n")
            f.write(f"{diff['128_resps']}\n\n")
            
            f.write("-" * 100 + "\n")
            f.write("FILTERED RESPONSES & EXACT MATCH:\n")
            f.write("-" * 100 + "\n")
            f.write(f"256 step filtered: {diff['256_filtered_resps']}\n")
            f.write(f"256 step exact_match: {diff['256_exact_match']}\n\n")
            f.write(f"128 step filtered: {diff['128_filtered_resps']}\n")
            f.write(f"128 step exact_match: {diff['128_exact_match']}\n\n")
    
    print(f"Found {len(differences)} cases with different responses")
    return len(differences)

if __name__ == "__main__":
    base_dir = Path("/workspace/Dream/eval_instruct/output_reproduce/gsm8k_realbaseline/Dream-org__Dream-v0-Instruct-7B")
    
    file_256 = base_dir / "256step.jsonl"
    file_128 = base_dir / "128step.jsonl"
    output_file = base_dir / "comparison_256_vs_128.txt"
    
    count = compare_files(file_256, file_128, output_file)
    
    print(f"\n✓ Comparison complete!")
    print(f"✓ Total cases with different responses: {count}")
    print(f"✓ Output saved to: {output_file}")
