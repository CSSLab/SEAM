from graph_tasks.path_cnt_task import main as path_count_task   
from graph_tasks.shortest_path_task import main as shortest_path_task
from graph_tasks.path_exist_task import main as path_exist_task
from graph_tasks.bfs_traversal_task import main as bfs_traversal_task
from graph_tasks.config import *
import argparse
import os


def task_path_counting(cfg):
    path_count_task(generation_cnt=cfg.samples, benchmark_root=cfg.benchmark_root)

def task_shortest_path(cfg):
    shortest_path_task(generation_cnt=cfg.samples, benchmark_root=cfg.benchmark_root)

def task_path_existence(cfg):
    path_exist_task(generation_cnt=cfg.samples, benchmark_root=cfg.benchmark_root)

def task_bfs_traversal(cfg):
    bfs_traversal_task(generation_cnt=cfg.samples, benchmark_root=cfg.benchmark_root)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_root", default="../data/benchmark", type=str)
    parser.add_argument("--samples", default=200, type=int)
    parser.add_argument("--offset", default=1, type=int)

    return parser.parse_args()

if __name__ == '__main__':
    cfg = parse_args()
    print("Configurations:", flush=True)
    for arg in vars(cfg):
        print(f"\t{arg}: {getattr(cfg, arg)}", flush=True)

    if not os.path.exists(cfg.benchmark_root):
        os.makedirs(cfg.benchmark_root)

    task_path_counting(cfg)
    task_path_existence(cfg)
    task_shortest_path(cfg)
    task_bfs_traversal(cfg)