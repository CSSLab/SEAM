from datasets import load_dataset
import pandas as pd
from music21 import environment, converter, layout
import os
from tqdm import tqdm
import json
import random
from PIL import Image
import argparse
import re
import io
import math
import numpy as np

# Environment setup for musescore
environLocal = environment.Environment()
environLocal['musescoreDirectPNGPath'] = '/usr/bin/musescore3'
environLocal['musicxmlPath'] = '/usr/bin/musescore3'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
# PNG_DPI = 300

def parse_abc_safely(abc_text):
    """Parse ABC notation safely"""
    try:
        # 1. Basic validation
        if not abc_text or not isinstance(abc_text, str):
            return None
            
        # 2. Check for key changes first - reject if found as it potentially hinders re matching
        if '[K:' in abc_text:
            return None
        
        header_lines = []
        music_lines = []
        original_meter = None
        
        for line in abc_text.split('\n'):
            if line.startswith(('X:', 'T:', 'L:', 'K:')):
                header_lines.append(line)
            elif line.startswith('M:'):
                if original_meter is None:
                    original_meter = line[2:].strip()
                header_lines.append(line)
            else:
                music_lines.append(line)
                
        if not original_meter:
            return None
            
        # 3. Check for problematic notation patterns
        music_content = ' '.join(music_lines)
        
        # Check for implicit meter changes
        if '[M:' in music_content:
            return None
            
        # Check for problematic patterns
        if any(pattern in music_content for pattern in [
            # Chord symbols that might be confused with note durations
            '[A-G]\\d{5,}\"',
            # Multiple consecutive section markers
            ':|:|',
            ':||:',
            # Multiple key changes
            '[K:.*][K:',
            # Multiple line breaks
            '\\n\\n\\n',
            # Grace notes inside chords
            '{.*}\\[',
            '\\[.*{.*}',
            # Problematic duration patterns
            '[A-Ga-g]\\d{7,}',
        ]):
            return None
            
        # Clean up certain notations that often cause issues
        cleaned_content = (
            music_content
            .replace('\\n\\n', '\\n')  # Normalize line breaks
        )
        
        # Reconstruct ABC notation
        cleaned_abc = '\n'.join(header_lines + [cleaned_content])
        
        # 4. Try parsing
        score = converter.parse(cleaned_abc, format='abc')
        
        # 5. Post-parse validation
        # Check measures exist
        if not score.parts or not score.parts[0].getElementsByClass('Measure'):
            return None
            
        # Verify meter consistency
        if not all(ts.ratioString == original_meter 
                  for ts in score.flat.getTimeSignatures()):
            return None
            
        # Verify reasonable total duration
        if score.duration.quarterLength <= 0 or score.duration.quarterLength > 1000:
            return None
        
        return score
        
    except Exception as e:
        return None

def count_specific_note(score, note_name):
    count = 0
    for note in score.flat.getElementsByClass('Note'):
        if note.pitch.step == note_name:
            count += 1
    return count

def get_all_measures(abc_text):
    lines = [line for line in abc_text.split('\n') 
            if not line.startswith(('X:', 'T:', 'M:', 'L:', 'K:'))]
    
    if not lines:
        return []
    
    measures = []
    for line in lines:
        parts = re.split(r':{1,2}\|{0,2}:{0,2}|\|{2}', line)
        
        for part in parts:
            subparts = part.split('|')
            
            for measure in subparts:
                cleaned = re.sub(r'^\d', '', measure)
                cleaned = cleaned.strip()
                if cleaned:
                    measures.append(cleaned)
    
    return measures

def get_rhythm_patterns(default_length):
    """
    Return patterns for four types of dotted notes based on default length.
    L:1/4 - no modifier = quarter note
    L:1/8 - no modifier = eighth note
    L:1/16 - no modifier = sixteenth note
    """
    if default_length == "1/4":
        # Default = quarter note
        dotted_sixteenth_patterns = [
            r'(?<!")[A-Ga-g]3/8(?!\])',  # explicit dotted sixteenth (exclude inside quotes)
            r'(?<!")[A-Ga-g]/4>',        # alternative notation (exclude inside quotes, d>)
            r'(?<!")<[A-Ga-g]/4'         # grace note followed by dotted note (exclude inside quotes, <d)
        ]
        
        dotted_eighth_patterns = [
            r'(?<!")[A-Ga-g]3/4(?!\])',  # explicit dotted eighth (exclude inside quotes)
            r'(?<!")[A-Ga-g]/2>',        # alternative notation (exclude inside quotes, d>)
            r'(?<!")<[A-Ga-g]/2'         # grace note followed by dotted note (exclude inside quotes, <d)
        ]
        
        dotted_quarter_patterns = [
            r'(?<!")[A-Ga-g]3/2(?!\])',  # explicit dotted quarter (exclude inside quotes)
            r'(?<!")[A-Ga-g]>',          # alternative notation (exclude inside quotes, d>)
            r'(?<!")<[A-Ga-g](?!\d)'     # grace note followed by dotted note (exclude inside quotes, <d)
        ]
        
        dotted_half_patterns = [
            r'(?<!")[A-Ga-g]3(?!/|\])',  # explicit dotted half (exclude inside quotes)
            r'(?<!")[A-Ga-g]2>',         # alternative notation (exclude inside quotes, d>)
            r'(?<!")<[A-Ga-g]2'          # grace note followed by dotted note (exclude inside quotes, <d)
        ]
        
    elif default_length == "1/8":
        # Default = eighth note
        dotted_sixteenth_patterns = [
            r'(?<!")[A-Ga-g]3/4(?!\])',  # explicit dotted sixteenth (exclude inside quotes)
            r'(?<!")[A-Ga-g]/2>',        # alternative notation (exclude inside quotes, d>)
            r'(?<!")<[A-Ga-g]/2'         # grace note followed by dotted note (exclude inside quotes, <d)
        ]
        
        dotted_eighth_patterns = [
            r'(?<!")[A-Ga-g]3/2(?!\])',  # explicit dotted eighth (exclude inside quotes)
            r'(?<!")[A-Ga-g]>',          # alternative notation (exclude inside quotes, d>)
            r'(?<!")<[A-Ga-g](?!\d)'     # grace note followed by dotted note (exclude inside quotes, <d)
        ]
        
        dotted_quarter_patterns = [
            r'(?<!")[A-Ga-g]3(?!/|\])',  # explicit dotted quarter (exclude inside quotes)
            r'(?<!")[A-Ga-g]2>',         # alternative notation (exclude inside quotes, d>)
            r'(?<!")<[A-Ga-g]2'          # grace note followed by dotted note (exclude inside quotes, <d)
        ]
        
        dotted_half_patterns = [
            r'(?<!")[A-Ga-g]6(?!/|\])',  # explicit dotted half (exclude inside quotes)
            r'(?<!")[A-Ga-g]4>',         # alternative notation (exclude inside quotes, d>)
            r'(?<!")<[A-Ga-g]4'          # grace note followed by dotted note (exclude inside quotes, <d)
        ]
        
    elif default_length == "1/16":
        # Default = sixteenth note
        dotted_sixteenth_patterns = [
            r'(?<!")[A-Ga-g]3/2(?!\])',  # explicit dotted sixteenth (exclude inside quotes)
            r'(?<!")[A-Ga-g]>',          # alternative notation (exclude inside quotes, d>)
            r'(?<!")<[A-Ga-g](?!\d)'     # grace note followed by dotted note (exclude inside quotes, <d)
        ]
        
        dotted_eighth_patterns = [
            r'(?<!")[A-Ga-g]3(?!/|\])',  # explicit dotted eighth (exclude inside quotes)
            r'(?<!")[A-Ga-g]2>',         # alternative notation (exclude inside quotes, d>)
            r'(?<!")<[A-Ga-g]2'          # grace note followed by dotted note (exclude inside quotes, <d)
        ]
        
        dotted_quarter_patterns = [
            r'(?<!")[A-Ga-g]6(?!/|\])',  # explicit dotted quarter (exclude inside quotes)
            r'(?<!")[A-Ga-g]4>',         # alternative notation (exclude inside quotes, d>)
            r'(?<!")<[A-Ga-g]4'          # grace note followed by dotted note (exclude inside quotes, <d)
        ]
        
        dotted_half_patterns = [
            r'(?<!")[A-Ga-g]12(?!/|\])',  # explicit dotted half (exclude inside quotes)
            r'(?<!")[A-Ga-g]8>',          # alternative notation (exclude inside quotes, d>)
            r'(?<!")<[A-Ga-g]8'           # grace note followed by dotted note (exclude inside quotes, <d)
        ]
    
    else:
        raise ValueError(f"Unsupported default length: {default_length}")
        
    return (dotted_sixteenth_patterns, dotted_eighth_patterns, 
            dotted_quarter_patterns, dotted_half_patterns)

def identify_rhythm_pattern(measure, pattern_type, default_length="1/8"):
    if default_length not in ["1/4", "1/8", "1/16"]:
        return False
        
    patterns = get_rhythm_patterns(default_length)
    pattern_dict = {
        "dotted_sixteenth": patterns[0],
        "dotted_eighth": patterns[1],
        "dotted_quarter": patterns[2],
        "dotted_half": patterns[3]
    }
    
    if pattern_type not in pattern_dict:
        raise ValueError(f"Unknown rhythm pattern type: {pattern_type}")
        
    return any(bool(re.search(pattern, measure)) for pattern in pattern_dict[pattern_type])

def create_options(correct_count):
    offset = 3
    if correct_count <= 2:
        correct_idx = 0
    elif correct_count <= 5:
        correct_idx = random.randint(0, 1)
    elif correct_count <= 8:
        correct_idx = random.randint(0, 2)
    else:
        correct_idx = random.randint(0, 3)

    options = []
    for i in range(4):
        if i < correct_idx:
            options.append(correct_count - (offset * (correct_idx - i)))
        elif i > correct_idx:
            options.append(correct_count + (offset * (i - correct_idx)))
        else:
            options.append(correct_count)
    
    new_options = options.copy()
    random.shuffle(new_options)
    new_correct_idx = new_options.index(correct_count)
    
    return new_options, new_correct_idx

def auto_crop_image(image, threshold=245):
    """
    Automatically crop the white space around an image.
    
    Args:
        image: PIL Image object
        threshold: Pixel value threshold for considering as background (0-255)
        
    Returns:
        Cropped PIL Image
    """
    # Convert image to numpy array
    img_data = np.array(image)
    
    # If image is RGB, convert to grayscale for threshold detection
    if len(img_data.shape) == 3:
        gray_data = np.mean(img_data, axis=2)
    else:
        gray_data = img_data
    
    # Find the bounding box of non-background pixels
    non_empty_columns = np.where(np.min(gray_data, axis=0) < threshold)[0]
    non_empty_rows = np.where(np.min(gray_data, axis=1) < threshold)[0]
    
    if len(non_empty_rows) == 0 or len(non_empty_columns) == 0:
        # Image is empty or all background
        return image
        
    # Add some padding (20px) around the content
    padding = 20
    cropBox = (
        max(0, min(non_empty_columns) - padding),
        max(0, min(non_empty_rows) - padding),
        min(gray_data.shape[1] - 1, max(non_empty_columns) + padding),
        min(gray_data.shape[0] - 1, max(non_empty_rows) + padding)
    )
    
    return image.crop(cropBox)

def score_to_square_image(score, output_path, size=(400, 400), bg_color='white'):
    """
    Convert a music21 score to a square image with proper layout, cropping, and formatting.
    
    Args:
        score: music21 score object
        output_path: Path where the final image will be saved
        size: Tuple of (width, height) for the final image size
        bg_color: Background color for the image
        measures_per_system: Number of measures per system (row)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        temp_file = output_path
        temp_path = f"{output_path.rsplit('.', 1)[0]}-1.png"  # musescore adds -1 to the filename
        
        # Generate the initial image with higher DPI for better clarity
        score.write('musicxml.png', fp=temp_file)
        
        # Open the PNG and convert to RGB if needed
        original_img = Image.open(temp_path)
        
        if original_img.mode != 'RGB':
            rgb_img = Image.new('RGB', original_img.size, bg_color)
            rgb_img.paste(original_img, mask=original_img.split()[3] if len(original_img.split()) > 3 else None)
            original_img = rgb_img
        
        # Auto-crop the image to remove excessive white space
        cropped_img = auto_crop_image(original_img)
        
        # Calculate dimensions for padding to make it square
        orig_width, orig_height = cropped_img.size
        max_dim = max(orig_width, orig_height)
        
        # Create a new square image
        square_img = Image.new('RGB', (max_dim, max_dim), bg_color)
        
        # Center the score in the square
        x_offset = (max_dim - orig_width) // 2
        y_offset = (max_dim - orig_height) // 2
        square_img.paste(cropped_img, (x_offset, y_offset))
        
        # Resize to desired output size (maintaining square aspect ratio)
        output_img = square_img.resize(size, Image.LANCZOS)
        
        # Save the final image
        output_img.save(output_path)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return True
        
    except Exception as e:
        print(f"Error creating square image: {e}")
        return False

def filter_by_measure_count(score, min_measures=40, max_measures=60):
    """Check if a score has an appropriate number of measures."""
    if not score or not score.parts:
        return False
    
    measure_count = len(score.parts[0].getElementsByClass('Measure'))
    return min_measures <= measure_count <= max_measures

def task_count_notes(cfg):
    if not os.path.exists(cfg.benchmark_root + "/notes"):
        os.makedirs(cfg.benchmark_root + "/notes")
    
    ret = []
    dataset = load_dataset("sander-wood/irishman", cache_dir="../data/music")
    df = pd.DataFrame(dataset['train'])
    
    for idx in tqdm(range(len(df))):
        abc_text = df.iloc[idx]['abc notation']
        try:
            score = parse_abc_safely(abc_text)
            if score is None or not filter_by_measure_count(score, cfg.min_measures, cfg.max_measures):
                continue
                
            # Save the sheet music as a square image
            output_path = f'{cfg.benchmark_root}/notes/{len(ret)}.png'
            if not score_to_square_image(score, output_path, 
                                       size=(cfg.image_size, cfg.image_size)):
                continue

            note_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
            note_counts = {}
            for note_name in note_names:
                count = count_specific_note(score, note_name)
                note_counts[note_name] = count

            valid_notes = [n for n, c in note_counts.items() if c > 0]
            if not valid_notes:
                continue
                
            target_note = random.choice(valid_notes)
            correct_count = note_counts[target_note]
            
            options, correct_idx = create_options(correct_count)
            
            ret.append({
                "abc_notation": abc_text,
                "target_note": target_note,
                "options": options,
                "correct_idx": correct_idx
            })
            
            if len(ret) == cfg.samples:
                break
                
        except Exception as e:
            continue
    
    with open(os.path.join(cfg.benchmark_root, "notes.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")

def task_count_measures(cfg):
    if not os.path.exists(cfg.benchmark_root + "/measures"):
        os.makedirs(cfg.benchmark_root + "/measures")
    
    ret = []
    dataset = load_dataset("sander-wood/irishman", cache_dir="../data/music")
    df = pd.DataFrame(dataset['train'])
    
    for idx in tqdm(range(len(df))):
        abc_text = df.iloc[idx]['abc notation']
        try:
            score = parse_abc_safely(abc_text)
            if score is None:
                continue
                
            part = score.parts[0]
            measures = [m for m in part.getElementsByClass('Measure')]
            measure_count = len(measures)

            if not filter_by_measure_count(score, cfg.min_measures, cfg.max_measures):
                continue
            
            # Save the sheet music as a square image
            output_path = f'{cfg.benchmark_root}/measures/{len(ret)}.png'
            if not score_to_square_image(score, output_path, 
                                       size=(cfg.image_size, cfg.image_size)):
                continue
            
            options, correct_idx = create_options(measure_count)
            
            ret.append({
                "abc_notation": abc_text,
                "options": options,
                "correct_idx": correct_idx
            })
            
            if len(ret) == cfg.samples:
                break
                
        except Exception as e:
            continue
    
    with open(os.path.join(cfg.benchmark_root, "measures.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")

def task_musical_form(cfg):
    if not os.path.exists(cfg.benchmark_root + "/forms"):
        os.makedirs(cfg.benchmark_root + "/forms")
    
    dataset = load_dataset("Seeker38/music_abc_notation_with_music_theory", cache_dir="../data/music")
    df = pd.DataFrame(dataset['train'])
    
    valid_forms = [
        "Only One Section", 
        "Through Composed", 
        "Compound Binary", 
        "Compound Ternary", 
        "American Popular"
    ]

    def extract_form(text):
        text = text.replace("Sectional: ", "").replace("Assistant: ", "").strip()
        for form in valid_forms:
            if form in text:
                return form
        return None

    form_examples = df[df['output'].apply(lambda x: extract_form(x) is not None)]
    form_counts = form_examples['output'].apply(extract_form).value_counts()
    available_forms = form_counts.index.tolist()[::-1]
    
    balanced_examples = []
    for form in available_forms:
        form_data = form_examples[form_examples['output'].apply(lambda x: extract_form(x) == form)]
        balanced_examples.append(form_data)
    combined_examples = pd.concat(balanced_examples)

    ret = []
    counter = {"Only One Section": 0, 
               "Through Composed": 0, 
               "Compound Binary": 0, 
               "Compound Ternary": 0, 
               "American Popular": 0}
    
    for _, row in tqdm(combined_examples.iterrows(), desc="Creating form tasks"):
        try:
            correct_form = extract_form(row['output'])
            
            if correct_form == "American Popular" or correct_form == "Compound Ternary":
                pass
            elif correct_form == "Compound Binary" or correct_form == "Through Composed":
                if counter[correct_form] >= 50:
                    continue
            else:
                if sum(counter.values()) >= cfg.samples:
                    break
            
            score = parse_abc_safely(row['input'])
            if score is None or not filter_by_measure_count(score, cfg.min_measures, cfg.max_measures):
                continue
                
            output_path = f'{cfg.benchmark_root}/forms/{len(ret)}.png'
            if not score_to_square_image(score, output_path, 
                                       size=(cfg.image_size, cfg.image_size)):
                continue
            
            other_forms = [f for f in available_forms if f != correct_form]
            wrong_options = random.sample(other_forms, 3)
            correct_idx = random.randint(0, 3)
            options = (wrong_options[:correct_idx] + [correct_form] + wrong_options[correct_idx:])
            
            ret.append({
                "abc_notation": row['input'],
                "options": options,
                "correct_idx": correct_idx
            })
            counter[correct_form] += 1
            
        except Exception as e:
            continue
    
    print(f"Successfully created {len(ret)} musical form examples")
    print(counter)
    
    with open(os.path.join(cfg.benchmark_root, "forms.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")

def task_rhythm_pattern(cfg):
    if not os.path.exists(cfg.benchmark_root + "/rhythm"):
        os.makedirs(cfg.benchmark_root + "/rhythm")
    
    ret = []
    dataset = load_dataset("sander-wood/irishman", cache_dir="../data/music")
    df = pd.DataFrame(dataset['train'])
    
    rhythm_types = ["dotted_sixteenth", "dotted_eighth", "dotted_quarter", "dotted_half"]
    samples_per_type = cfg.samples // len(rhythm_types)
    
    for rhythm_type in rhythm_types:
        examples_count = 0
        
        for idx in tqdm(range(len(df)), desc=f"Processing {rhythm_type}"):
            try:
                abc_text = df.iloc[idx]['abc notation']
                
                default_length = "1/8"  # Default value
                for line in abc_text.split('\n'):
                    if line.startswith('L:'):
                        default_length = line[2:].strip()
                        break
                if default_length not in ["1/4", "1/8", "1/16"]:
                    continue

                meter = None
                for line in abc_text.split('\n'):
                    if line.startswith('M:'):
                        meter = line[2:].strip()
                        break
                if meter not in ["4/4", "2/4", "3/4"]:
                    continue

                measures = get_all_measures(abc_text)
                
                pattern_measures = []
                for i, measure in enumerate(measures, 1):
                    if identify_rhythm_pattern(measure, rhythm_type, default_length):
                        pattern_measures.append(i)
                
                if not pattern_measures:
                    continue
                
                score = parse_abc_safely(abc_text)
                if score is None or not filter_by_measure_count(score, cfg.min_measures, cfg.max_measures):
                    continue
                    
                output_path = f'{cfg.benchmark_root}/rhythm/{len(ret)}.png'
                if not score_to_square_image(score, output_path, 
                                           size=(cfg.image_size, cfg.image_size)):
                    continue
                
                correct_measure = random.choice(pattern_measures)
                other_measures = [i for i in range(1, len(measures)+1) 
                                if i not in pattern_measures]
                
                if len(other_measures) < 3:
                    continue

                wrong_options = random.sample(other_measures, 3)
                correct_idx = random.randint(0, 3)
                options = (wrong_options[:correct_idx] + 
                          [correct_measure] + 
                          wrong_options[correct_idx:])
                
                ret.append({
                    "abc_notation": abc_text,
                    "rhythm_type": rhythm_type,
                    "options": options,
                    "correct_idx": correct_idx
                })
                
                examples_count += 1
                if examples_count >= samples_per_type:
                    break
                    
            except Exception as e:
                continue
    
    print(f"Successfully created {len(ret)} rhythm pattern examples")
    
    with open(os.path.join(cfg.benchmark_root, "rhythm.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_root", default="../data/benchmark", type=str)
    parser.add_argument("--samples", default=200, type=int)
    parser.add_argument("--min_measures", default=24, type=int, 
                      help="Minimum number of measures for filtering")
    parser.add_argument("--max_measures", default=48, type=int, 
                      help="Maximum number of measures for filtering")
    parser.add_argument("--image_size", default=600, type=int, 
                      help="Size of the square output images")
    return parser.parse_args()

if __name__ == "__main__":
    cfg = parse_args()
    print("Configurations:", flush=True)
    for arg in vars(cfg):
        print(f"\t{arg}: {getattr(cfg, arg)}", flush=True)
    
    if not os.path.exists(cfg.benchmark_root):
        os.makedirs(cfg.benchmark_root)
    
    task_count_notes(cfg)
    task_count_measures(cfg)
    task_musical_form(cfg)
    task_rhythm_pattern(cfg)