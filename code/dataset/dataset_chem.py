import os
import io
import json
import pdb
import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from PIL import Image
import random
from rdkit.Chem import Descriptors
from utils.utils import hash_str
from datasets import load_dataset
import tqdm
import ast
import math
from rdkit.Geometry import Point3D
from vllm import LLM
import torch

def smiles_to_image(smiles, size=(400, 400), rotation_angle=0):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Generate 2D coordinates for the molecule
    AllChem.Compute2DCoords(mol)
    
    # Rotate the molecule by modifying the 2D coordinates
    conf = mol.GetConformer()
    center_x = sum([conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]) / mol.GetNumAtoms()
    center_y = sum([conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]) / mol.GetNumAtoms()
    
    rot_rad = math.radians(rotation_angle)
    cos_rot = math.cos(rot_rad)
    sin_rot = math.sin(rot_rad)
    
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        x = pos.x - center_x
        y = pos.y - center_y
        
        # Apply rotation
        new_x = x * cos_rot - y * sin_rot + center_x
        new_y = x * sin_rot + y * cos_rot + center_y
        
        conf.SetAtomPosition(i, Point3D(new_x, new_y, 0))
    
    # Draw the molecule and return PIL Image
    img = Draw.MolToImage(mol, size=size)
    
    return img


def sample_options(correct_idx, correct, all_options, embs, num_options=4):

    correct_emb = embs[all_options.index(correct)]
    sims = torch.nn.functional.cosine_similarity(embs, correct_emb.unsqueeze(dim=0), dim=1)
    top_sims, top_idxs = torch.topk(sims, k=num_options+1)
    
    options = [all_options[idx] for idx in top_idxs[1:]]
    options.insert(correct_idx, correct)
    
    assert correct == options[correct_idx]
    
    return options


def task_carbon(cfg, source_datset):
    
    if not os.path.exists(os.path.join(cfg.benchmark_root, "carbon")):
        os.makedirs(os.path.join(cfg.benchmark_root, "carbon"))
        
    if not os.path.exists(os.path.join(cfg.benchmark_root, "carbon_rot")):
        os.makedirs(os.path.join(cfg.benchmark_root, "carbon_rot"))

    data = []
    for item in tqdm.tqdm(source_datset["test"]):
        if item["question"] == "What are the most likely reactants that can produce the following organic molecule in a chemical reaction?":
            smiles = item["description"].strip(".").split("Its SMILES notation is ")[1]
            mol = Chem.MolFromSmiles(smiles)
            c_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
            if c_count > 10:
                data.append({"smiles": smiles, "c_count": c_count})
            if len(data) == cfg.samples:
                break
    
    data = pd.DataFrame(data)
    
    ret = []
    for i, row in data.iterrows():
        smiles = row["smiles"]
        count = row["c_count"]
        
        correct_idx = random.randint(0, 3)
        options = []
        for i in range(4):
            offset = abs(i - correct_idx) * cfg.carbon_offset
            if i < correct_idx:
                options.append(count - offset)
            elif i > correct_idx:
                options.append(count + offset)
            else:
                options.append(count)
        
        ret.append({"smiles": smiles,
                    "options": options,
                    "correct_idx": correct_idx})
        
        img = smiles_to_image(smiles, size=(400, 400), rotation_angle=0)
        filename = f"{hash_str(smiles)}.png"
        img.save(os.path.join(cfg.benchmark_root, "carbon", filename))
        
        img = smiles_to_image(smiles, size=(400, 400), rotation_angle=180)
        filename = f"{hash_str(smiles)}.png"
        img.save(os.path.join(cfg.benchmark_root, "carbon_rot", filename))

    assert len(ret) == cfg.samples
    
    print(ret[-1])
    # {'smiles': 'Cc1nn(C)cc1CN1CCN(c2nccnc2Cl)CC1', 'options': [11, 14, 17, 20], 'correct_idx': 3}

    with open(os.path.join(cfg.benchmark_root, "carbon.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")

    with open(os.path.join(cfg.benchmark_root, "carbon_rot.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")
            

def task_hydrogen(cfg, source_datset):
    
    if not os.path.exists(os.path.join(cfg.benchmark_root, "hydrogen")):
        os.makedirs(os.path.join(cfg.benchmark_root, "hydrogen"))
    
    if not os.path.exists(os.path.join(cfg.benchmark_root, "hydrogen_rot")):
        os.makedirs(os.path.join(cfg.benchmark_root, "hydrogen_rot"))
    
    data = []
    for item in tqdm.tqdm(source_datset["test"]):
        if item["question"] == "What are the most likely reactants that can produce the following organic molecule in a chemical reaction?":
            smiles = item["description"].strip(".").split("Its SMILES notation is ")[1]
            mol = Chem.MolFromSmiles(smiles)
            h_count = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
            if h_count > 10:
                data.append({"smiles": smiles, "h_count": h_count})
            if len(data) == cfg.samples:
                break
    
    data = pd.DataFrame(data)
    
    ret = []
    for i, row in data.iterrows():
        smiles = row["smiles"]
        count = row["h_count"]
        
        correct_idx = random.randint(0, 3)
        options = []
        for i in range(4):
            offset = abs(i - correct_idx) * cfg.hydrogen_offset
            if i < correct_idx:
                options.append(count - offset)
            elif i > correct_idx:
                options.append(count + offset)
            else:
                options.append(count)
        
        ret.append({"smiles": smiles,
                    "options": options,
                    "correct_idx": correct_idx})
        
        img = smiles_to_image(smiles, size=(400, 400), rotation_angle=0)
        filename = f"{hash_str(smiles)}.png"
        img.save(os.path.join(cfg.benchmark_root, "hydrogen", filename))
        
        img = smiles_to_image(smiles, size=(400, 400), rotation_angle=180)
        filename = f"{hash_str(smiles)}.png"
        img.save(os.path.join(cfg.benchmark_root, "hydrogen_rot", filename))
        
    assert len(ret) == cfg.samples
    
    print(ret[-1])
    # {'smiles': 'Cc1nn(C)cc1CN1CCN(c2nccnc2Cl)CC1', 'options': [19, 20, 21, 22], 'correct_idx': 0}

    with open(os.path.join(cfg.benchmark_root, "hydrogen.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")
    
    with open(os.path.join(cfg.benchmark_root, "hydrogen_rot.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")


def task_weight(cfg, source_datset):
    
    if not os.path.exists(os.path.join(cfg.benchmark_root, "weight")):
        os.makedirs(os.path.join(cfg.benchmark_root, "weight"))
    
    if not os.path.exists(os.path.join(cfg.benchmark_root, "weight_rot")):
        os.makedirs(os.path.join(cfg.benchmark_root, "weight_rot"))
        
    data = []
    for item in tqdm.tqdm(source_datset["test"]):
        if item["question"] == "What are the most likely reactants that can produce the following organic molecule in a chemical reaction?":
            smiles = item["description"].strip(".").split("Its SMILES notation is ")[1]
            mol = Chem.MolFromSmiles(smiles)
            weight = round(Descriptors.MolWt(mol))
            data.append({"smiles": smiles, "weight": weight})
            if len(data) == cfg.samples:
                break
    
    data = pd.DataFrame(data)
    
    ret = []
    for i, row in tqdm.tqdm(data.iterrows()):
        smiles = row["smiles"]
        weight = row["weight"]
        
        correct_idx = random.randint(0, 3)
        
        options = []
        for i in range(4):
            offset = round(abs(i - correct_idx) * cfg.weight_offset * weight)
            if i < correct_idx:
                options.append(weight - offset)
            elif i > correct_idx:
                options.append(weight + offset)
            else:
                options.append(weight)
        
        ret.append({"smiles": smiles,
                    "options": options,
                    "correct_idx": correct_idx})
        
        img = smiles_to_image(smiles, size=(400, 400), rotation_angle=0)
        filename = f"{hash_str(smiles)}.png"
        img.save(os.path.join(cfg.benchmark_root, "weight", filename))
        
        img = smiles_to_image(smiles, size=(400, 400), rotation_angle=180)
        filename = f"{hash_str(smiles)}.png"
        img.save(os.path.join(cfg.benchmark_root, "weight_rot", filename))
    
    assert len(ret) == cfg.samples
    
    print(ret[-1])
    # {'smiles': 'Cc1nn(C)cc1CN1CCN(c2nccnc2Cl)CC1', 'options': [296.8, 306.8, 316.8, 326.8], 'correct_idx': 1}

    with open(os.path.join(cfg.benchmark_root, f"weight.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")
    
    with open(os.path.join(cfg.benchmark_root, f"weight_rot.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")


def get_embs(cfg, prompts):
    
    model = LLM(model=os.path.join(cfg.model_root, cfg.emb_model),
                enforce_eager=True)
    outputs = model.encode(prompts)
    
    embs = torch.cat([output.outputs.data.unsqueeze(dim=0) for output in outputs], dim=0)
    
    return embs

def task_caption(cfg, source_datset):
    
    
    if not os.path.exists(os.path.join(cfg.benchmark_root, "caption")):
        os.makedirs(os.path.join(cfg.benchmark_root, "caption"))
    
    if not os.path.exists(os.path.join(cfg.benchmark_root, "caption_rot")):
        os.makedirs(os.path.join(cfg.benchmark_root, "caption_rot"))
        
    caption = []
    for item in tqdm.tqdm(source_datset["test"]):
        if item["question"] == "What is the most relevant description of the following organic molecule?":
            pos = ast.literal_eval(item["choices"])[item["label"]]
            smiles = item["description"].strip(".").split("Its SMILES notation is ")[1]
            caption.append({"smiles": smiles, "caption": pos})
    
    data = pd.DataFrame(caption)
    all_captions = data["caption"].tolist()
    embs = get_embs(cfg, all_captions)
    
    ret = []
    for i, row in tqdm.tqdm(data.iterrows()):
        smiles = row["smiles"]
        caption = row["caption"]
        correct_idx = random.randint(0, 3)
        
        options = sample_options(correct_idx, caption, all_captions, embs)
        
        ret.append({"smiles": smiles,
                    "options": options,
                    "correct_idx": correct_idx})

        img = smiles_to_image(smiles, size=(400, 400), rotation_angle=0)
        filename = f"{hash_str(smiles)}.png"
        img.save(os.path.join(cfg.benchmark_root, "caption", filename))
        
        img = smiles_to_image(smiles, size=(400, 400), rotation_angle=180)
        filename = f"{hash_str(smiles)}.png"
        img.save(os.path.join(cfg.benchmark_root, "caption_rot", filename))
        
        if len(ret) == cfg.samples:
            break
    
    assert len(ret) == cfg.samples
    
    print(ret[-1])
    # {'smiles': 'CC(CC(=O)C(=O)[O-])(C(=O)[O-])O', 'options': ['The molecule is the dicarboxylic acid dianion resulting from deprotonation of both carboxy groups of 4-hydroxy-4-methyl-2-oxoglutaric acid; major species at pH 7.3. It is a dicarboxylic acid dianion and an oxo carboxylic acid anion. It is a conjugate base of a 4-hydroxy-4-methyl-2-oxoglutaric acid.', 'The molecule is a heterodetic cyclic peptide that is biosynthesised by an engineered strain of Escherichia coli It has a role as an Escherichia coli metabolite. It is a heterodetic cyclic peptide, a dithioacetal and a monohydroxyquinoline.', "The molecule is a flavanone glycoside that is (2S)-flavanone substituted by hydroxy groups at positions 5, 2' and 5', methyl groups at positions 6 and 8 and a (6''-O-p-hydroxybenzoyl)-beta-D-glucopyranosyloxy residue at position 7. Isolated from the leaves of Myrcia multiflora, it exhibits inhibitory activity against aldose reductase. It has a role as an EC 1.1.1.21 (aldehyde reductase) inhibitor and a plant metabolite. It is a beta-D-glucoside, a flavanone glycoside, a monosaccharide derivative, a trihydroxyflavanone and a 4-hydroxybenzoate ester.", 'The molecule is a phenylacetaldehyde in which the 3 and 4 positions of the phenyl group are substituted by hydroxy groups. It has a role as a human metabolite, an Escherichia coli metabolite and a mouse metabolite. It is a member of catechols, an alpha-CH2-containing aldehyde and a member of phenylacetaldehydes.'], 'correct_idx': 0}
    
    with open(os.path.join(cfg.benchmark_root, f"caption.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")
    
    with open(os.path.join(cfg.benchmark_root, f"caption_rot.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_root", default="/data/models/benchmark_models", type=str)
    parser.add_argument("--benchmark_root", default="../data/benchmark", type=str)
    parser.add_argument("--data_path", default="../data/chem/llm_test.csv", type=str)
    parser.add_argument("--emb_model", default="multilingual-e5-large-instruct", type=str)
    parser.add_argument("--caption_path", default="../data/chem/molecule_captioning_test.csv", type=str)
    parser.add_argument("--samples", default=200, type=int)
    parser.add_argument("--source_data", default="shangzhu/ChemQA", type=str)
    parser.add_argument("--cache_dir", default="../data/chem/ChemQA", type=str)
    parser.add_argument("--weight_offset", default=0.2, type=float)
    parser.add_argument("--carbon_offset", default=3, type=int)
    parser.add_argument("--hydrogen_offset", default=3, type=int)

    return parser.parse_args()


if __name__ == "__main__":

    cfg = parse_args()
    print("Configurations:", flush=True)
    for arg in vars(cfg):
        print(f"\t{arg}: {getattr(cfg, arg)}", flush=True)

    dataset = load_dataset(cfg.source_data, cache_dir=cfg.cache_dir)
    
    if not os.path.exists(cfg.benchmark_root):
        os.makedirs(cfg.benchmark_root)

    task_carbon(cfg, dataset)
    task_hydrogen(cfg, dataset)
    task_weight(cfg, dataset)
    task_caption(cfg, dataset)
    
    
