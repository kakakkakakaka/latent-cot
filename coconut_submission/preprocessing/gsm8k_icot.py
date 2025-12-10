# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Preprocessing script for GSM8K dataset with iCoT enhancement (controllable size).
Downloads iCoT-enhanced GSM8K data and converts to coconut training format.
Supports sampling to control the final dataset size (default: 70k-100k samples).
"""

import json
import argparse
import os
import re
import random
from pathlib import Path
import requests
from tqdm import tqdm


def parse_answer(answer_text):
    """
    Parse GSM8K answer text to extract reasoning steps and final answer.
    
    iCoT data format typically has more detailed steps than original GSM8K.
    """
    if not answer_text:
        return [], ""
    
    # Try to find the final answer marked with #### (GSM8K format)
    final_answer_match = re.search(r'####\s*([^\n]+)', answer_text)
    if final_answer_match:
        final_answer = final_answer_match.group(1).strip()
        # Remove the final answer from the text to get steps
        steps_text = answer_text[:final_answer_match.start()].strip()
    else:
        # Try to find answer after "Therefore" or similar phrases
        therefore_match = re.search(r'(?:Therefore|So|The answer is|Answer:|Final answer:)\s*([^\n.]+)', answer_text, re.IGNORECASE)
        if therefore_match:
            final_answer = therefore_match.group(1).strip()
            steps_text = answer_text[:therefore_match.start()].strip()
        else:
            # If no clear marker, try to extract the last number or sentence
            lines = answer_text.strip().split('\n')
            if len(lines) > 1:
                final_answer = lines[-1].strip()
                steps_text = '\n'.join(lines[:-1]).strip()
            else:
                # Fallback: use everything as steps, extract any number as answer
                numbers = re.findall(r'\d+', answer_text)
                final_answer = numbers[-1] if numbers else ""
                steps_text = answer_text
    
    # GSM8K format: each line is typically a reasoning step
    # Steps are separated by newlines, with calculator annotations like <<calculation>>
    lines = [l.strip() for l in steps_text.split('\n') if l.strip()]
    
    # Each line is a reasoning step, but we should clean up calculator annotations
    cleaned_steps = []
    for line in lines:
        # Remove calculator annotations like <<48/2=24>> 
        cleaned_line = re.sub(r'<<[^>]+>>', '', line)
        cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()
        if cleaned_line:
            cleaned_steps.append(cleaned_line)
    
    # If no steps found after cleaning, use original lines
    if not cleaned_steps:
        cleaned_steps = [l.strip() for l in lines if l.strip()]
    
    # If still no steps, use the full text as one step
    if not cleaned_steps and steps_text:
        cleaned_steps = [steps_text.strip()]
    
    # Clean final answer - remove $, commas, and extra whitespace
    final_answer = re.sub(r'[$,]', '', final_answer).strip()
    
    return cleaned_steps, final_answer


def download_icot_data(output_dir="data", split="train"):
    """
    Download iCoT-enhanced GSM8K data from the repository.
    
    iCoT data is available at: https://github.com/da03/Internalize_CoT_Step_by_Step
    We'll download from HuggingFace datasets or the raw files.
    """
    print(f"Downloading iCoT-enhanced GSM8K {split} data...")
    
    try:
        from datasets import load_dataset
        print("Attempting to load from HuggingFace datasets...")
        
        # Try to load iCoT data from HuggingFace (if available)
        # Note: This may not be available, so we'll provide alternative method
        try:
            dataset = load_dataset("da03/GSM8K_Internalize_CoT", split=split)
            print(f"Loaded {len(dataset)} examples from HuggingFace")
            return dataset
        except:
            print("HuggingFace dataset not available, trying alternative method...")
            
            # Alternative: Download from GitHub raw files
            return download_from_github(output_dir, split)
            
    except ImportError:
        print("datasets library not available, downloading from GitHub...")
        return download_from_github(output_dir, split)


def download_from_github(output_dir, split):
    """
    Download iCoT data from GitHub repository.
    iCoT data location: https://github.com/da03/Internalize_CoT_Step_by_Step/tree/main/data/gsm8k
    
    Note: The files use Git LFS, so we need to use the actual LFS download URL or parse the .txt format.
    """
    # The actual files are .txt format (not .json) and use Git LFS
    # Format: question||reasoning_steps #### answer (one per line)
    
    if split == "train":
        filename = "train.txt"
        output_file = os.path.join(output_dir, "gsm8k_train_icot_raw.txt")
    else:  # validation/test
        filename = "test.txt"  # or "valid.txt"
        output_file = os.path.join(output_dir, "gsm8k_validation_icot_raw.txt")
    
    # Try using huggingface_hub to download from da03/GSM8K_Internalize_CoT
    # Or use direct GitHub download
    base_urls = [
        "https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/main/data/gsm8k",
    ]
    
    # First try GitHub media URL (for Git LFS files)
    media_url = f"https://media.githubusercontent.com/media/da03/Internalize_CoT_Step_by_Step/refs/heads/main/data/gsm8k/{filename}"
    try:
        print(f"Attempting to download from GitHub media (for LFS files): {media_url}")
        response = requests.get(media_url, timeout=60, stream=True)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_file, 'wb') as f, tqdm(
            desc=f"Downloading {filename}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        
        print(f"Downloaded {filename} from GitHub media, parsing...")
        return parse_txt_file(output_file, split)
        
    except Exception as e:
        print(f"Failed to download from GitHub media: {e}")
        print("Trying alternative URL...")
    
    # Fallback: Try raw.githubusercontent.com (may be LFS pointer)
    for base_url in base_urls:
        url = f"{base_url}/{filename}"
        try:
            print(f"Trying to download: {url}")
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Save to file first
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Check if it's a Git LFS pointer file
            with open(output_file, 'r', encoding='utf-8') as f:
                first_200 = f.read(200)
                if 'version https://git-lfs.github.com/spec/v1' in first_200:
                    print(f"Detected Git LFS pointer file. Attempting to download actual content...")
                    # Try to download from GitHub LFS
                    return download_git_lfs_file(output_dir, split, filename, first_200)
            
            # If not LFS, parse the .txt file directly
            print(f"Downloaded {filename}, parsing...")
            return parse_txt_file(output_file, split)
            
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            continue
    
    # Alternative: Try to use huggingface_hub if available
    try:
        from huggingface_hub import hf_hub_download
        print("Attempting to download from HuggingFace using huggingface_hub...")
        repo_id = "da03/GSM8K_Internalize_CoT"
        try:
            # Try different possible file names
            hf_filename = f"gsm8k_{split}.txt" if split == "train" else f"gsm8k_{split}.txt"
            downloaded_file = hf_hub_download(repo_id=repo_id, filename=hf_filename, local_dir=output_dir)
            return parse_txt_file(downloaded_file, split)
        except:
            # Try alternative names
            hf_filename = f"{split}.txt"
            try:
                downloaded_file = hf_hub_download(repo_id=repo_id, filename=hf_filename, local_dir=output_dir)
                return parse_txt_file(downloaded_file, split)
            except Exception as e2:
                print(f"Failed to download from HuggingFace: {e2}")
    except ImportError:
        print("huggingface_hub not available. Install with: pip install huggingface_hub")
    
    raise FileNotFoundError(
        f"Could not download iCoT data for {split} split.\n"
        f"Please download manually from: https://github.com/da03/Internalize_CoT_Step_by_Step\n"
        f"Or install git-lfs and clone the repository:\n"
        f"  git lfs install\n"
        f"  git clone https://github.com/da03/Internalize_CoT_Step_by_Step.git"
    )


def download_git_lfs_file(output_dir, split, filename, lfs_pointer):
    """Attempt to download actual Git LFS file content."""
    # Parse LFS pointer to get oid
    import re
    oid_match = re.search(r'oid sha256:([a-f0-9]+)', lfs_pointer)
    if not oid_match:
        raise ValueError("Could not parse Git LFS pointer")
    
    oid = oid_match.group(1)
    print(f"Git LFS OID: {oid}")
    
    # Try GitHub LFS API
    # GitHub LFS uses: https://github.com/<owner>/<repo>.git/info/lfs/objects/batch
    # For raw content: https://media.githubusercontent.com/media/<owner>/<repo>/refs/heads/<branch>/<path>
    
    base_url = "https://media.githubusercontent.com/media/da03/Internalize_CoT_Step_by_Step/refs/heads/main/data/gsm8k"
    url = f"{base_url}/{filename}"
    
    try:
        print(f"Trying GitHub media URL: {url}")
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()
        
        output_file = os.path.join(output_dir, f"gsm8k_{split}_icot_raw.txt")
        with open(output_file, 'wb') as f, tqdm(
            desc=filename,
            total=int(response.headers.get('content-length', 0)),
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        
        print(f"Downloaded LFS file to: {output_file}")
        return parse_txt_file(output_file, split)
    except Exception as e:
        print(f"Failed to download from GitHub media: {e}")
        raise


def parse_txt_file(txt_file, split):
    """
    Parse iCoT .txt file format.
    Format: question||reasoning_steps #### answer (one per line)
    """
    print(f"Parsing {txt_file}...")
    data = []
    
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Skip Git LFS pointer lines
            if line.startswith('version https://git-lfs.github.com'):
                continue
            if line.startswith('oid sha256:') or line.startswith('size '):
                continue
            
            # Parse format: question||reasoning_steps #### answer
            if '||' in line and '####' in line:
                parts = line.split('||', 1)
                if len(parts) == 2:
                    question = parts[0].strip()
                    answer_part = parts[1].strip()
                    
                    # Split reasoning and final answer
                    if '####' in answer_part:
                        reasoning_parts = answer_part.split('####', 1)
                        reasoning = reasoning_parts[0].strip()
                        final_answer = reasoning_parts[1].strip()
                    else:
                        reasoning = answer_part
                        final_answer = ""
                    
                    # Convert to coconut format
                    # Split reasoning into steps (by <<>> markers or newlines)
                    steps = []
                    if '<<' in reasoning:
                        # Format with <<step>> markers
                        import re
                        step_pattern = r'<<([^>]+)>>'
                        matches = re.findall(step_pattern, reasoning)
                        if matches:
                            steps = [m.strip() for m in matches]
                    else:
                        # Split by lines or use whole reasoning as one step
                        lines = [l.strip() for l in reasoning.split('\n') if l.strip()]
                        steps = lines if lines else [reasoning] if reasoning else []
                    
                    # Ensure we have at least one step
                    if not steps:
                        steps = [reasoning] if reasoning else [""]
                    
                    data.append({
                        "question": question,
                        "steps": steps,
                        "answer": final_answer
                    })
                    
                    if len(data) % 10000 == 0:
                        print(f"  Parsed {len(data)} examples...")
    
    print(f"Parsed {len(data)} examples from {txt_file}")
    return data


def download_file(url, output_file):
    """Download a file from URL with progress bar (for JSON files)."""
    print(f"Downloading from: {url}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    # Download file
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    with open(output_file, 'wb') as f, tqdm(
        desc=os.path.basename(output_file),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    
    print(f"Downloaded to: {output_file}")
    
    # Load JSON file (handle both JSON and JSONL formats)
    data = []
    with open(output_file, 'r', encoding='utf-8') as f:
        if output_file.endswith('.jsonl'):
            # JSONL format: one JSON object per line
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        else:
            # Regular JSON format
            data = json.load(f)
    
    # If data is a dict, try to extract list
    if isinstance(data, dict):
        # Try common keys
        for key in ['data', 'examples', 'samples', 'train', 'test']:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    
    return data
    
    # Load JSON file
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def convert_icot_to_coconut_format(data, output_dir="data", target_size=None, split="train"):
    """
    Convert iCoT-enhanced GSM8K dataset to coconut training format.
    Optionally sample to control the dataset size.
    
    Args:
        data: List of iCoT examples or dataset object
        output_dir: Output directory for JSON files
        target_size: Target number of samples (None = use all, or int for sampling)
        split: Dataset split ('train' or 'validation')
    """
    # Convert to list if it's a dataset object
    if hasattr(data, '__iter__') and not isinstance(data, (list, tuple)):
        data = list(data)
    
    print(f"Converting {len(data)} iCoT examples to coconut format...")
    
    coconut_data = []
    
    for idx, example in enumerate(tqdm(data, desc="Processing")):
        # Handle different possible formats
        if isinstance(example, dict):
            question = example.get("question", example.get("input", ""))
            answer_text = example.get("answer", example.get("output", ""))
            # IMPORTANT: If example already has "steps" field (from parse_txt_file),
            # use it directly instead of re-parsing from answer_text
            existing_steps = example.get("steps", None)
        else:
            # If it's a string or other format, skip
            continue
        
        if not question:
            continue
        
        # If steps are already parsed (from parse_txt_file), use them directly
        if existing_steps and isinstance(existing_steps, list) and len(existing_steps) > 0:
            steps = existing_steps
            # Use the answer from the example, or extract from answer_text if needed
            if answer_text:
                final_answer = answer_text
            else:
                # Extract answer from last step or use last number
                numbers = re.findall(r'\d+', steps[-1] if steps else "")
                final_answer = numbers[-1] if numbers else (steps[-1] if steps else "")
        else:
            # No existing steps, parse from answer_text
            if not answer_text:
                continue
            
            # Parse answer to get steps and final answer
            steps, final_answer = parse_answer(answer_text)
            
            # If we couldn't parse properly, use a simple approach
            if not steps:
                steps = [answer_text]
                numbers = re.findall(r'\d+', answer_text)
                if numbers:
                    final_answer = numbers[-1]
                else:
                    final_answer = answer_text.split()[-1] if answer_text.split() else ""
        
        # Ensure we have at least one step
        if not steps:
            steps = [answer_text] if answer_text else [""]
        
        # Ensure we have a final answer
        if not final_answer:
            if answer_text:
                numbers = re.findall(r'\d+', answer_text)
                final_answer = numbers[-1] if numbers else answer_text.split()[-1] if answer_text.split() else ""
            else:
                # Extract from last step
                numbers = re.findall(r'\d+', steps[-1] if steps else "")
                final_answer = numbers[-1] if numbers else ""
        
        coconut_example = {
            "question": question,
            "steps": steps,
            "answer": final_answer
        }
        
        coconut_data.append(coconut_example)
    
    print(f"Converted {len(coconut_data)} examples")
    
    # Sample if target_size is specified
    # IMPORTANT: For Stage 1 (Coconut) training, we need multiple CoT reasoning 
    # for the same question. So we preserve multiple samples per question.
    if target_size and len(coconut_data) > target_size:
        print(f"Sampling {target_size} examples from {len(coconut_data)} total examples")
        print("Using question-aware sampling to preserve multiple CoT variants per question...")
        
        # Group examples by question to ensure multiple CoT per question
        from collections import defaultdict
        question_groups = defaultdict(list)
        for example in coconut_data:
            question_groups[example["question"]].append(example)
        
        print(f"Found {len(question_groups)} unique questions")
        print(f"Average {len(coconut_data) / len(question_groups):.1f} samples per question")
        
        # Check if we have multiple variants per question
        # iCoT data may have mostly unique questions (1 variant each) or multiple variants per question
        avg_variants = len(coconut_data) / len(question_groups)
        
        if avg_variants < 1.5:
            # Most questions have only 1 variant - iCoT data is question expansion, not CoT expansion
            print(f"⚠️  Warning: Average {avg_variants:.2f} variants per question.")
            print(f"   Most questions have only 1 variant. This means iCoT data expands the question set,")
            print(f"   not the CoT variants per question.")
            print(f"   Using random sampling instead of question-aware sampling.")
            
            # Random sampling
            random.seed(42)
            sampled_data = random.sample(coconut_data, min(target_size, len(coconut_data)))
            coconut_data = sampled_data
        else:
            # Multiple variants per question - use question-aware sampling
            print(f"✓ Multiple variants per question detected (avg: {avg_variants:.2f})")
            print(f"  Using question-aware sampling to preserve multiple CoT per question...")
            
            # Calculate how many samples to keep per question
            # Strategy: Keep 5-8 samples per question for Stage 1 (Coconut needs multiple CoT per question)
            min_samples_per_question = 6  # Minimum 6 samples per question (5-8 range)
            max_samples_per_question = 8  # Maximum 8 samples per question
            total_unique_questions = len(question_groups)
            
            # Sort questions by number of variants (descending) for better distribution
            sorted_questions = sorted(
                question_groups.items(), 
                key=lambda x: len(x[1]), 
                reverse=True
            )
            
            # Calculate target samples per question based on target_size
            min_total_needed = total_unique_questions * min_samples_per_question
            max_total_possible = total_unique_questions * max_samples_per_question
            
            samples_per_question_dict = {}
            
            if target_size < min_total_needed:
                # Target too small, use minimum for all (or as many as available)
                print(f"Warning: Target size {target_size} is smaller than minimum needed {min_total_needed}")
                print(f"Using minimum {min_samples_per_question} samples per question (or as many as available)")
                for question, examples in question_groups.items():
                    samples_per_question_dict[question] = min(min_samples_per_question, len(examples))
            elif target_size <= max_total_possible:
                # Target fits within 5-8 range
                avg_samples = target_size / total_unique_questions
                print(f"Target size allows {avg_samples:.2f} samples per question (within 5-8 range)")
                
                base_allocation = int(avg_samples)
                extra_needed = target_size - (base_allocation * total_unique_questions)
                
                # First, give base allocation to all
                for question, examples in question_groups.items():
                    samples_per_question_dict[question] = min(base_allocation, len(examples), max_samples_per_question)
                
                # Distribute extra samples
                for question, examples in sorted_questions:
                    if extra_needed <= 0:
                        break
                    current = samples_per_question_dict[question]
                    if current < max_samples_per_question and current < len(examples):
                        add = min(extra_needed, max_samples_per_question - current, len(examples) - current)
                        samples_per_question_dict[question] = current + add
                        extra_needed -= add
            else:
                # Target exceeds max (8 per question)
                print(f"Warning: Target size {target_size} exceeds maximum {max_total_possible}")
                print(f"Using maximum {max_samples_per_question} samples per question")
                remaining = target_size
                for question, examples in sorted_questions:
                    n = min(max_samples_per_question, len(examples), remaining)
                    samples_per_question_dict[question] = n
                    remaining -= n
            
            # Sample from each question group
            random.seed(42)
            sampled_data = []
            for question, examples in question_groups.items():
                n_samples = samples_per_question_dict.get(question, min_samples_per_question)
                n_samples = min(n_samples, len(examples))  # Don't exceed available
                sampled_examples = random.sample(examples, n_samples)
                sampled_data.extend(sampled_examples)
            
            coconut_data = sampled_data
        print(f"Final dataset size: {len(coconut_data)} examples")
        
        # Verify: Check sample distribution per question
        question_counts = defaultdict(int)
        for example in coconut_data:
            question_counts[example["question"]] += 1
        
        counts = list(question_counts.values())
        print(f"Average {sum(counts)/len(counts):.1f} samples per question")
        print(f"Range: {min(counts)}-{max(counts)} samples per question")
        print(f"Questions with 5-8 samples: {sum(1 for c in counts if 5 <= c <= 8)}/{len(counts)} ({100*sum(1 for c in counts if 5 <= c <= 8)/len(counts):.1f}%)")
        questions_with_multiple = sum(1 for count in counts if count >= 2)
        print(f"Questions with multiple CoT variants (≥2): {questions_with_multiple}/{len(question_counts)} ({100*questions_with_multiple/len(question_counts):.1f}%)")
    
    # Save to JSON file
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"gsm8k_{split}.json" if split != "test" else "gsm8k_validation.json"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coconut_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(coconut_data)} examples to {output_path}")
    print(f"Sample example:")
    if coconut_data:
        print(f"  Question: {coconut_data[0]['question'][:100]}...")
        print(f"  Steps: {len(coconut_data[0]['steps'])} steps")
        print(f"  Answer: {coconut_data[0]['answer']}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert iCoT-enhanced GSM8K dataset to coconut training format with controllable size."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to process"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for JSON files"
    )
    parser.add_argument(
        "--target-size",
        type=lambda x: None if x.lower() == 'none' else int(x),
        default=None,
        help="Target number of samples (for train split, recommended: 70000-100000). 'None' = use all data"
    )
    parser.add_argument(
        "--from-file",
        type=str,
        default=None,
        help="Load iCoT data from local JSON file instead of downloading"
    )
    args = parser.parse_args()
    
    # Load data
    if args.from_file:
        print(f"Loading iCoT data from local file: {args.from_file}")
        with open(args.from_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = download_icot_data(args.output_dir, args.split)
    
    # Convert and save
    convert_icot_to_coconut_format(
        data, 
        args.output_dir, 
        target_size=args.target_size if args.split == "train" else None,
        split=args.split
    )


if __name__ == "__main__":
    main()

