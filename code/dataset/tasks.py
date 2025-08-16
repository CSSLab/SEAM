import json
import re
import pdb
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.utils import (
    read_image_chess,
    read_image_chem,
    read_image_music,
    read_image_graph
)


suffix = (
    "Let's think step-by-step to answer the above question.\n"
    "One and only one option is correct. If you are unsure, provide your best guess.\n"
    "If you believe none of the options are correct, select the closest one.\n"
    "You MUST conclude with: The best option is [the_option_letter].\n"
    "where the [the_option_letter] MUST be one of A, B, C or D.\n"
)


def format_options(options):

    prompt = f"A. {options[0]}\n"
    prompt += f"B. {options[1]}\n"
    prompt += f"C. {options[2]}\n"
    prompt += f"D. {options[3]}\n\n"
    
    return prompt

def get_prefix(cfg, notation, notation_name):
    
    if cfg.mode == "l" or cfg.mode == "vl":
        notation_text = f"{notation_name}: {notation}\n\n"
    elif cfg.mode == "v":
        notation_text = ""
    else:
        raise ValueError("Invalid mode")
    
    return notation_text


def format_task_fork(line, cfg):
    
    fen, answer = line["fen"], chr(line["options"].index(line["correct_square"]) + 65)
    instructions = (
        "Question: Which of the following pieces is forking other pieces in this chess position?\n"
        "A forking piece is one that attacks two or more enemy pieces simultaneously.\n"
    )
    prompt = get_prefix(cfg, fen, "FEN") + instructions + format_options(line["options"]) + suffix
    
    return prompt, answer, fen


def format_task_legal(line, cfg):
    
    fen, answer = line["fen"], chr(line["legal_move_idx"] + 65)
    instructions = "Question: Which of the following moves is legal in this chess position?\n"
    prompt = get_prefix(cfg, fen, "FEN") + instructions + format_options(line["options_uci"]) + suffix
    
    return prompt, answer, fen


def format_task_puzzle(line, cfg):
    
    fen, answer = line["fen"], chr(line["best_move_idx"] + 65)
    instructions = "Question: Which of the following moves is the best next move for the active player in this chess position?\n"
    prompt = get_prefix(cfg, fen, "FEN") + instructions + format_options(line["options_san"]) + suffix
    
    return prompt, answer, fen


def format_task_eval(line, cfg):
    
    fen, answer = line["fen"], chr(line["correct_idx"] + 65)

    instructions = (
        "Question: Which of the following is the correct centipawn evaluation for this chess position?\n"
        "The evaluation shows the overall positional strength (including piece activity, king safety, and other strategic factors), not just the material count and piece value advantage.\n"
        "Positive numbers mean White has an advantage, negative numbers mean Black has an advantage, and 0 means the position is equal.\n"
    )

    prompt = get_prefix(cfg, fen, "FEN") + instructions + format_options(line["options"]) + suffix
    
    return prompt, answer, fen


def format_task_carbon(line, cfg):
    
    smiles, answer = line["smiles"], chr(line["correct_idx"] + 65)
    instructions = "Question: Which of the following is the correct number of carbon atoms in this compound?\n"
    prompt = get_prefix(cfg, smiles, "SMILES") + instructions + format_options(line["options"]) + suffix
    
    return prompt, answer, smiles


def format_task_hydrogen(line, cfg):
    
    smiles, answer = line["smiles"], chr(line["correct_idx"] + 65)
    instructions = "Question: Which of the following is the correct number of hydrogen atoms in this compound?\n"
    prompt = get_prefix(cfg, smiles, "SMILES") + instructions + format_options(line["options"]) + suffix
    
    return prompt, answer, smiles


def format_task_weight(line, cfg):
    
    smiles, answer = line["smiles"], chr(line["correct_idx"] + 65)
    instructions = "Question: Which of the following is the correct molecular weight of this compound?\n"
    prompt = get_prefix(cfg, smiles, "SMILES") + instructions + format_options(line["options"]) + suffix
    
    return prompt, answer, smiles

def format_task_caption(line, cfg):
    
    smiles, answer = line["smiles"], chr(line["correct_idx"] + 65)
    instructions = "Question: Which of the following descriptions is correct for this compound?\n"
    prompt = get_prefix(cfg, smiles, "SMILES") + instructions + format_options(line["options"]) + suffix
    
    return prompt, answer, smiles


def format_task_notes(line, cfg):
    
    abc, answer = line["abc_notation"], chr(line["correct_idx"] + 65)
    
    instructions = (
        f"Question: How many individual {line['target_note']} notes (including {line['target_note']}♯, "
        f"{line['target_note']}♭, and {line['target_note']}♮) appear in this piece? Count all occurrences "
        f"of {line['target_note']} regardless of whether they are sharp, flat, or natural, but do not count "
        f"notes that appear as part of chords.\n"
    )
    prompt = get_prefix(cfg, abc, "ABC Notation") + instructions + format_options(line["options"]) + suffix

    return prompt, answer

def format_task_measures(line, cfg):
    
    abc, answer = line["abc_notation"], chr(line["correct_idx"] + 65)
    instructions = (
        "Question: Which of the following is the correct number of measures in this piece?\n"
        "Count each measure only once, ignoring any repetition signs in the score.\n"
    )
    prompt = get_prefix(cfg, abc, "ABC Notation") + instructions + format_options(line["options"]) + suffix

    return prompt, answer

def format_task_forms(line, cfg):
    
    abc, answer = line["abc_notation"], chr(line["correct_idx"] + 65)
    instructions = "Question: Which of the following best describes the musical form of this piece?\n"
    prompt = get_prefix(cfg, abc, "ABC Notation") + instructions + format_options(line["options"]) + suffix
    
    return prompt, answer
    

def format_task_rhythm(line, cfg):
    
    abc, answer = line["abc_notation"], chr(line["correct_idx"] + 65)
    pattern_names = {
        "dotted_sixteenth": "dotted sixteenth note",
        "dotted_eighth": "dotted eighth note",
        "dotted_quarter": "dotted quarter note",
        "dotted_half": "dotted half note"
    }
    
    instructions = f"Question: Which of the following measures contains a {pattern_names[line['rhythm_type']]}?\n"
    prompt = get_prefix(cfg, abc, "ABC Notation") + instructions + format_options(line["options"]) + suffix
    
    return prompt, answer


def format_task_path_counting(line, cfg):
    
    adj_matrix, answer = line["matrix"], chr(line["correct_idx"] + 65)
    source_node, target_node = line["source_node"], line["target_node"]
    instructions = (
        f"Question: Which of the following is the correct number of unique simple paths from {source_node} to {target_node} in the graph?\n"
    )
    prompt = get_prefix(cfg, adj_matrix, "Adjacency matrix") + instructions + format_options(line["options"]) + suffix
    
    return prompt, answer


def format_task_path_existence(line, cfg):
    
    adj_matrix, answer = line["matrix"], chr(line["correct_idx"] + 65)
    source_node, target_node = line["source_node"], line["target_node"]
    instructions = (
        f"Question: Which of the following node lists represents a path from {source_node} to {target_node} in the graph?\n"
    )
    prompt = get_prefix(cfg, adj_matrix, "Adjacency matrix") + instructions + format_options(line["options"]) + suffix
    
    return prompt, answer

def format_task_shortest_path(line, cfg):
    
    adj_matrix, answer = line["matrix"], chr(line["correct_idx"] + 65)
    source_node, target_node = line["source_node"], line["target_node"]
    instructions = (
        f"Question: Which of the following is the length of the shortest simple path from {source_node} to {target_node} in the graph?\n"
    )
    prompt = get_prefix(cfg, adj_matrix, "Adjacency matrix") + instructions + format_options(line["options"]) + suffix
    
    return prompt, answer


def format_task_bfs_traversal(line, cfg):
    
    adj_matrix, answer = line["matrix"], chr(line["correct_idx"] + 65)
    start_node = line['start_node']
    instructions = (
        f"Question: Which of the following node lists represents the order of the BFS traversal starting from node {start_node} in the graph?\n"
    )
    prompt = get_prefix(cfg, adj_matrix, "Adjacency matrix") + instructions + format_options(line["options"]) + suffix
    
    return prompt, answer

# rewrite this function
def load_tasks(cfg):
    
    chess_tasks = ["fork", "legal", "puzzle", "eval"]
    chem_tasks = ["carbon", "hydrogen", "weight", "caption"]
    music_tasks = ["notes", "measures", "forms", "rhythm"]
    graph_tasks = ["path_counting", "path_existence", "shortest_path", "bfs_traversal"]
    
    chess_res_tasks = ["fork_res", "legal_res", "puzzle_res", "eval_res"]
    chess_bw_tasks = ["fork_bw", "legal_bw", "puzzle_bw", "eval_bw"]
    chem_rot_tasks = ["carbon_rot", "hydrogen_rot", "weight_rot", "caption_rot"]
    
    tasks_dict = {
        "chess": chess_tasks,
        "chem": chem_tasks,
        "music": music_tasks,
        "graph": graph_tasks,
        "chess_res": chess_res_tasks,
        "chess_bw": chess_bw_tasks,
        "chem_rot": chem_rot_tasks
    }
    
    if cfg.tasks == "bench":
        tasks = chess_tasks + chem_tasks + music_tasks + graph_tasks
    elif cfg.tasks == "variant":
        tasks = chess_res_tasks + chess_bw_tasks + chem_rot_tasks
    elif cfg.tasks in tasks_dict:
        tasks = tasks_dict[cfg.tasks]
    else:
        tasks = cfg.tasks.split(",")
    
    print(tasks)
    
    music_tasks_dict = {
        "notes": format_task_notes,
        "measures": format_task_measures,
        "forms": format_task_forms,
        "rhythm": format_task_rhythm
    }
    
    graph_tasks_dict = {
        "path_counting": format_task_path_counting,
        "path_existence": format_task_path_existence,
        "shortest_path": format_task_shortest_path,
        "bfs_traversal": format_task_bfs_traversal
    }
    
    ret = {}
    for task in tasks:
    
        questions = []
        answers = []
        images = []
        
        with open(cfg.benchmark_root + task + ".jsonl", "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                line = json.loads(line)
                if task in ["fork", "fork_res", "fork_bw"]:
                    prompt, answer, fen = format_task_fork(line, cfg)
                    img = read_image_chess(cfg, fen, task)
                elif task in ["legal", "legal_res", "legal_bw"]:
                    prompt, answer, fen = format_task_legal(line, cfg)
                    img = read_image_chess(cfg, fen, task)
                elif task in ["puzzle", "puzzle_res", "puzzle_bw"]:
                    prompt, answer, fen = format_task_puzzle(line, cfg)
                    img = read_image_chess(cfg, fen, task)
                elif task in ["eval", "eval_res", "eval_bw"]:
                    prompt, answer, fen = format_task_eval(line, cfg)
                    img = read_image_chess(cfg, fen, task)
                elif task in ["carbon", "carbon_rot"]:
                    prompt, answer, smiles = format_task_carbon(line, cfg)
                    img = read_image_chem(cfg, smiles, task)
                elif task in ["hydrogen", "hydrogen_rot"]:
                    prompt, answer, smiles = format_task_hydrogen(line, cfg)
                    img = read_image_chem(cfg, smiles, task)
                elif task in ["weight", "weight_rot"]:
                    prompt, answer, smiles = format_task_weight(line, cfg)
                    img = read_image_chem(cfg, smiles, task)
                elif task in ["caption", "caption_rot"]:
                    prompt, answer, smiles = format_task_caption(line, cfg)
                    img = read_image_chem(cfg, smiles, task)
                elif task in music_tasks_dict:
                    prompt, answer = music_tasks_dict[task](line, cfg)
                    img = read_image_music(cfg, idx, task)
                elif task in graph_tasks_dict:
                    prompt, answer = graph_tasks_dict[task](line, cfg)
                    img = read_image_graph(cfg, idx, task)
                else:
                    raise ValueError("Invalid task")
                
                questions.append(prompt)
                answers.append(answer)
                images.append(img)
                
        ret[task] = {"questions": questions, "answers": answers, "images": images}
    
    return ret
    