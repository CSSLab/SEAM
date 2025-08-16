import os
import io
import json
import pdb
import argparse
import tqdm
import chess
import chess.engine
import chess.pgn
import chess.svg
import random
from PIL import Image
import cairosvg

def fen2img(fen, size=400, bw=False):
    # Create a board from FEN
    board = chess.Board(fen)
    
    # Determine if it's black's turn
    is_black_turn = not board.turn
    
    # Define colors based on bw parameter
    if bw:
        colors = {
            'square light': '#FFFFFF',  # White squares
            'square dark': '#808080',   # Gray squares
            'margin': '#000000',        # Black border
            'coord': '#808080'          # Black coordinates
        }
    else:
        colors = {
            'square light': '#F0D9B5',  # Light brown squares
            'square dark': '#B58863',   # Dark brown squares
            'margin': '#000000',        # Black border
            'coord': '#808080'          # Black coordinates
        }
    
    # Generate SVG string
    svg_string = chess.svg.board(
        board=board,
        size=size,
        coordinates=True,  # Show coordinates around the board
        flipped=is_black_turn,  # Flip board when it's black's turn
        colors=colors
    )
    
    # Convert SVG to PNG using cairosvg
    png_data = cairosvg.svg2png(
        bytestring=svg_string.encode('utf-8'),
        output_width=size,
        output_height=size
    )
    
    # Create PIL Image from PNG data
    image = Image.open(io.BytesIO(png_data))
    
    return image

def task_pin(cfg):
    
    if not os.path.exists(os.path.join(cfg.benchmark_root, "pin")):
        os.makedirs(os.path.join(cfg.benchmark_root, "pin"))
    
    if not os.path.exists(os.path.join(cfg.benchmark_root, "pin_res")):
        os.makedirs(os.path.join(cfg.benchmark_root, "pin_res"))
    
    if not os.path.exists(os.path.join(cfg.benchmark_root, "pin_bw")):
        os.makedirs(os.path.join(cfg.benchmark_root, "pin_bw"))

    ret = []
    with open(cfg.puzzle_path, 'r') as f:
        f.readline()
        for line in tqdm.tqdm(f):
            line = line.strip().split(',')
            fen, moves = line[1], line[2]
            pre_move = moves.split(' ')[0]
            board = chess.Board(fen)
            board.push(chess.Move.from_uci(pre_move))
            fen = board.fen()
            
            pinned_pieces = []
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and board.is_pinned(piece.color, square):
                    pinned_pieces.append(square)
            
            if len(pinned_pieces):
                correct_square = random.choice(pinned_pieces)
                other_squares = [sq for sq in chess.SQUARES
                               if board.piece_at(sq)
                               and sq not in pinned_pieces]
                
                if len(other_squares) >= 3:
                    wrong_options = random.sample(other_squares, 3)
                    correct_idx = random.randint(0, 3)
                    options = (wrong_options[:correct_idx] +
                             [correct_square] +
                             wrong_options[correct_idx:])
                    
                    formatted_options = []
                    for sq in options:
                        piece = board.piece_at(sq)
                        color = "White" if piece.color else "Black"
                        piece_name = PIECE_NAMES[piece.symbol().lower()]
                        formatted_square = f"{color}'s {piece_name} on {chess.SQUARE_NAMES[sq]}"
                        formatted_options.append(formatted_square)
                    
                    img = fen2img(fen, size=400, bw=False)
                    file_name = fen.split(" ")[0].replace("/", "_") + ".png"
                    img.save(os.path.join(cfg.benchmark_root, "pin", file_name))
                    
                    img = fen2img(fen, size=300, bw=False)
                    file_name = fen.split(" ")[0].replace("/", "_") + ".png"
                    img.save(os.path.join(cfg.benchmark_root, "pin_res", file_name))
                    
                    img = fen2img(fen, size=400, bw=True)
                    file_name = fen.split(" ")[0].replace("/", "_") + ".png"
                    img.save(os.path.join(cfg.benchmark_root, "pin_bw", file_name))
                    
                    correct_piece = board.piece_at(correct_square)
                    correct_color = "White" if correct_piece.color else "Black"
                    correct_piece_name = PIECE_NAMES[correct_piece.symbol().lower()]
                    correct_formatted = f"{correct_color}'s {correct_piece_name} on {chess.SQUARE_NAMES[correct_square]}"
                    
                    ret.append({
                        "fen": fen,
                        "correct_square": correct_formatted,
                        "options": formatted_options
                    })
                    
                if len(ret) == cfg.samples:
                    break
    
    print(ret[-1])
    
    with open(os.path.join(cfg.benchmark_root, f"pin.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")

    with open(os.path.join(cfg.benchmark_root, f"pin_res.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")

    with open(os.path.join(cfg.benchmark_root, f"pin_bw.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")


def task_fork(cfg):

    if not os.path.exists(os.path.join(cfg.benchmark_root, "fork")):
        os.makedirs(os.path.join(cfg.benchmark_root, "fork"))
    
    if not os.path.exists(os.path.join(cfg.benchmark_root, "fork_res")):
        os.makedirs(os.path.join(cfg.benchmark_root, "fork_res"))
    
    if not os.path.exists(os.path.join(cfg.benchmark_root, "fork_bw")):
        os.makedirs(os.path.join(cfg.benchmark_root, "fork_bw"))
    
    ret = []
    with open(cfg.puzzle_path, 'r') as f:
        f.readline()
        for line in tqdm.tqdm(f):
            line = line.strip().split(',')
            fen, moves = line[1], line[2]
            pre_move = moves.split(' ')[0]
            board = chess.Board(fen)
            board.push(chess.Move.from_uci(pre_move))
            fen = board.fen()

            fork_squares = []
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if not piece:
                    continue
                    
                attacks = board.attacks(square)
                attacked_pieces = []
                
                for attack_square in attacks:
                    target = board.piece_at(attack_square)
                    if target and target.color != piece.color:
                        attacked_pieces.append(attack_square)
                
                if len(attacked_pieces) >= 2:
                    fork_squares.append(square)
            
            if len(fork_squares):
                correct_square = random.choice(fork_squares)
                other_squares = [sq for sq in chess.SQUARES 
                               if board.piece_at(sq) 
                               and sq not in fork_squares]
                
                if len(other_squares) >= 3:
                    wrong_options = random.sample(other_squares, 3)
                    correct_idx = random.randint(0, 3)
                    options = (wrong_options[:correct_idx] + 
                             [correct_square] + 
                             wrong_options[correct_idx:])
                    
                    formatted_options = []
                    for sq in options:
                        piece = board.piece_at(sq)
                        color = "White" if piece.color else "Black"
                        piece_name = PIECE_NAMES[piece.symbol().lower()]
                        formatted_square = f"{color}'s {piece_name} on {chess.SQUARE_NAMES[sq]}"
                        formatted_options.append(formatted_square)
                    
                    img = fen2img(fen, size=400, bw=False)
                    file_name = fen.split(" ")[0].replace("/", "_") + ".png"
                    img.save(os.path.join(cfg.benchmark_root, "fork", file_name))
                    
                    img = fen2img(fen, size=300, bw=False)
                    file_name = fen.split(" ")[0].replace("/", "_") + ".png"
                    img.save(os.path.join(cfg.benchmark_root, "fork_res", file_name))
                    
                    img = fen2img(fen, size=400, bw=True)
                    file_name = fen.split(" ")[0].replace("/", "_") + ".png"
                    img.save(os.path.join(cfg.benchmark_root, "fork_bw", file_name))
                    
                    correct_piece = board.piece_at(correct_square)
                    correct_color = "White" if correct_piece.color else "Black"
                    correct_piece_name = PIECE_NAMES[correct_piece.symbol().lower()]
                    correct_formatted = f"{correct_color}'s {correct_piece_name} on {chess.SQUARE_NAMES[correct_square]}"
                    
                    ret.append({
                        "fen": fen,
                        "correct_square": correct_formatted,
                        "options": formatted_options
                    })
            
            if len(ret) == cfg.samples:
                break
    
    print(ret[-1])
    
    with open(os.path.join(cfg.benchmark_root, f"fork.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")
    
    with open(os.path.join(cfg.benchmark_root, f"fork_res.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")
    
    with open(os.path.join(cfg.benchmark_root, f"fork_bw.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")


def generate_naive_illegal_moves(board):
    """Generate some naive illegal moves with the correct starting piece color."""
    naive_moves = set()
    all_squares = [chess.square_name(sq) for sq in range(64)]
    
    while len(naive_moves) < 10:  # Generate up to 10 naive illegal moves
        from_square = random.choice(all_squares)
        to_square = random.choice(all_squares)
        
        # Ensure the source square contains a piece of the correct color
        if from_square != to_square and board.piece_at(chess.parse_square(from_square)):
            piece = board.piece_at(chess.parse_square(from_square))
            if piece.color == board.turn:  # Check if the piece belongs to the side to move
                try:
                    move = chess.Move.from_uci(from_square + to_square)
                    # Ensure the move is not pseudo-legal or legal
                    if not board.is_pseudo_legal(move):
                        naive_moves.add(move)
                except chess.InvalidMoveError:
                    pass  # Ignore invalid moves that don't follow UCI format
    
    return naive_moves


def task_legal(cfg):
    
    if not os.path.exists(os.path.join(cfg.benchmark_root, "legal")):
        os.makedirs(os.path.join(cfg.benchmark_root, "legal"))
    
    if not os.path.exists(os.path.join(cfg.benchmark_root, "legal_res")):
        os.makedirs(os.path.join(cfg.benchmark_root, "legal_res"))
    
    if not os.path.exists(os.path.join(cfg.benchmark_root, "legal_bw")):
        os.makedirs(os.path.join(cfg.benchmark_root, "legal_bw"))
    
    ret = []
    with open(cfg.puzzle_path, 'r') as f:
        f.readline()  
        for line in tqdm.tqdm(f):
            line = line.strip().split(',')
            fen, moves = line[1], line[2]
            pre_move = moves.split(' ')[0]
            board = chess.Board(fen)
            board.push(chess.Move.from_uci(pre_move))
            
            fen = board.fen()
            
            pseudo_legal_moves = set(board.pseudo_legal_moves)
            legal_moves = set(board.legal_moves)
            false_moves = list(pseudo_legal_moves - legal_moves)
            naive_illegal_moves = generate_naive_illegal_moves(board)
            
            if legal_moves and false_moves and naive_illegal_moves:
                # Select one legal move
                option_legal = random.sample(list(legal_moves), 1)
                # Select one pseudo-legal but illegal move
                # option_pseudo_illegal = random.sample(false_moves, 1)
                # Select two naive illegal moves
                options_naive_illegal = random.sample(list(naive_illegal_moves), 3)
                
                # Combine options
                # options = option_pseudo_illegal + options_naive_illegal
                options = options_naive_illegal
                legal_move_idx = random.randint(0, 3)  # Random index for the legal move
                options.insert(legal_move_idx, option_legal[0])
                
                options_uci = [move.uci() for move in options]

                img = fen2img(fen, size=400, bw=False)
                file_name = fen.split(" ")[0].replace("/", "_") + ".png"
                img.save(os.path.join(cfg.benchmark_root, "legal", file_name))
                
                img = fen2img(fen, size=300, bw=False)
                file_name = fen.split(" ")[0].replace("/", "_") + ".png"
                img.save(os.path.join(cfg.benchmark_root, "legal_res", file_name))
                
                img = fen2img(fen, size=400, bw=True)
                file_name = fen.split(" ")[0].replace("/", "_") + ".png"
                img.save(os.path.join(cfg.benchmark_root, "legal_bw", file_name))

                ret.append({
                    "fen": fen,
                    "legal_move_uci": option_legal[0].uci(),
                    "legal_move_idx": legal_move_idx,
                    "options_uci": options_uci
                })

            if len(ret) == cfg.samples:
                break
        
        print(ret[-1])
    
    with open(os.path.join(cfg.benchmark_root, f"legal.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")
    
    with open(os.path.join(cfg.benchmark_root, f"legal_res.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")
    
    with open(os.path.join(cfg.benchmark_root, f"legal_bw.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")


def task_puzzle(cfg):
    
    if not os.path.exists(os.path.join(cfg.benchmark_root, "puzzle")):
        os.makedirs(os.path.join(cfg.benchmark_root, "puzzle"))
    
    if not os.path.exists(os.path.join(cfg.benchmark_root, "puzzle_res")):
        os.makedirs(os.path.join(cfg.benchmark_root, "puzzle_res"))
    
    if not os.path.exists(os.path.join(cfg.benchmark_root, "puzzle_bw")):
        os.makedirs(os.path.join(cfg.benchmark_root, "puzzle_bw"))
    
    ret = []
    with open(cfg.puzzle_path, 'r') as f:
        f.readline()
        for line in tqdm.tqdm(f):
            
            line = line.strip().split(',')
            fen, moves, elo = line[1], line[2], line[3]
            
            if int(elo) > cfg.max_elo:
                continue
            
            board = chess.Board(fen)
            moves = moves.split(' ')

            pre_move = chess.Move.from_uci(moves[0])
            board.push(pre_move)
            
            best_move_uci = moves[1]
            best_move_san = board.san(chess.Move.from_uci(best_move_uci))

            options_uci = [move.uci() for move in board.legal_moves if move.uci() != best_move_uci]
            options_san = [board.san(move) for move in board.legal_moves if move.uci() != best_move_uci]
            
            assert len(options_uci) == len(options_san)
            
            if len(options_uci) >= 3:
                options_uci = random.sample(options_uci, 3)
                options_san = random.sample(options_san, 3)
            
                best_move_idx = random.randint(0, 3)
                options_uci = options_uci[:best_move_idx] + [best_move_uci] + options_uci[best_move_idx:]
                options_san = options_san[:best_move_idx] + [best_move_san] + options_san[best_move_idx:]
                
                fen = board.fen()
                img = fen2img(fen, size=400, bw=False)
                file_name = fen.split(" ")[0].replace("/", "_") + ".png"
                img.save(os.path.join(cfg.benchmark_root, "puzzle", file_name))
                
                img = fen2img(fen, size=300, bw=False)
                file_name = fen.split(" ")[0].replace("/", "_") + ".png"
                img.save(os.path.join(cfg.benchmark_root, "puzzle_res", file_name))
                
                img = fen2img(fen, size=400, bw=True)
                file_name = fen.split(" ")[0].replace("/", "_") + ".png"
                img.save(os.path.join(cfg.benchmark_root, "puzzle_bw", file_name))
            
                ret.append({"fen": fen, 
                            "best_move_uci": best_move_uci,
                            "best_move_san": best_move_san,
                            "best_move_idx": best_move_idx,
                            "options_uci": options_uci,
                            "options_san": options_san,
                            "elo": elo})

            if len(ret) == cfg.samples:
                break
    
    with open(os.path.join(cfg.benchmark_root, f"puzzle.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")

    with open(os.path.join(cfg.benchmark_root, f"puzzle_res.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")
    
    with open(os.path.join(cfg.benchmark_root, f"puzzle_bw.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")


def task_eval(cfg):
    
    if not os.path.exists(os.path.join(cfg.benchmark_root, "eval")):
        os.makedirs(os.path.join(cfg.benchmark_root, "eval"))
    
    if not os.path.exists(os.path.join(cfg.benchmark_root, "eval_res")):
        os.makedirs(os.path.join(cfg.benchmark_root, "eval_res"))
    
    if not os.path.exists(os.path.join(cfg.benchmark_root, "eval_bw")):
        os.makedirs(os.path.join(cfg.benchmark_root, "eval_bw"))
    
    ret = []
    with open(cfg.eval_path, 'r') as f:
        for line in tqdm.tqdm(f):
            line = json.loads(line)
            fen = line['fen']
            depth = line['evals'][0]['depth']
            if depth >= 40:
                evals = line['evals'][0]['pvs']
                try:
                    if 500 >= abs(evals[0]['cp']) >= 200:
                        correct_cp = evals[0]['cp']
                        correct_idx = random.randint(0, 3)
                        options = []
                        for i in range(4):
                            # offset = round(abs(i - correct_idx) * correct_cp * cfg.cp_offset)
                            offset = abs(i - correct_idx) * cfg.cp_offset
                            if i < correct_idx:
                                options.append(correct_cp - offset)
                            elif i > correct_idx:
                                options.append(correct_cp + offset)
                            else:
                                options.append(correct_cp)
                        
                        ret.append({"fen": fen,
                                    "options": options,
                                    "correct_idx": correct_idx,
                                    "pv": evals[0]['line']})
                        
                        img = fen2img(fen, size=400, bw=False)
                        file_name = fen.split(" ")[0].replace("/", "_") + ".png"
                        img.save(os.path.join(cfg.benchmark_root, "eval", file_name))
                        
                        img = fen2img(fen, size=300, bw=False)
                        file_name = fen.split(" ")[0].replace("/", "_") + ".png"
                        img.save(os.path.join(cfg.benchmark_root, "eval_res", file_name))
                        
                        img = fen2img(fen, size=400, bw=True)
                        file_name = fen.split(" ")[0].replace("/", "_") + ".png"
                        img.save(os.path.join(cfg.benchmark_root, "eval_bw", file_name))
                        
                except:
                    pass
            if len(ret) == cfg.samples:
                break
    
    with open(os.path.join(cfg.benchmark_root, f"eval.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")
    
    with open(os.path.join(cfg.benchmark_root, f"eval_res.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")
    
    with open(os.path.join(cfg.benchmark_root, f"eval_bw.jsonl"), "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_root", default="../data/benchmark", type=str)
    parser.add_argument("--puzzle_path", default="../data/chess/lichess_db_puzzle.csv", type=str)
    parser.add_argument("--eval_path", default="../data/chess/lichess_db_eval_samples.jsonl", type=str)
    parser.add_argument("--samples", default=200, type=int)
    parser.add_argument("--cp_offset", default=300, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_elo", default=1200, type=int)

    return parser.parse_args()


if __name__ == "__main__":

    cfg = parse_args()
    print("Configurations:", flush=True)
    for arg in vars(cfg):
        print(f"\t{arg}: {getattr(cfg, arg)}", flush=True)

    PIECE_NAMES = {
        'p': 'Pawn',
        'n': 'Knight',
        'b': 'Bishop',
        'r': 'Rook',
        'q': 'Queen',
        'k': 'King'
    }
    
    if not os.path.exists(cfg.benchmark_root):
        os.makedirs(cfg.benchmark_root)
    
    task_fork(cfg)
    task_legal(cfg)
    task_puzzle(cfg)
    task_eval(cfg)
    
    
