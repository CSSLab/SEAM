#!/usr/bin/env python3
"""
SEAM Benchmark - Unified Metrics Computation
Computes comprehensive metrics from extracted results and generates comparison plots.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Local imports
from config.config import ALL_TASKS, ALL_MODES
from config.config import RESULTS_DIR

# Optional plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Domain mapping
DOMAIN_MAPPING = {
    # Chess tasks
    "fork": "chess", "legal": "chess", "puzzle": "chess", "eval": "chess",
    # Chemistry tasks  
    "carbon": "chemistry", "hydrogen": "chemistry", "weight": "chemistry", "caption": "chemistry",
    # Music tasks
    "notes": "music", "measures": "music", "forms": "music", "rhythm": "music",
    # Graph theory tasks
    "path_counting": "graph", "path_existence": "graph", "shortest_path": "graph", "bfs_traversal": "graph"
}

class MetricsCalculator:
    """Unified metrics calculator for SEAM benchmark results"""
    
    def __init__(self):
        self.domains = ["chess", "chemistry", "music", "graph"]
        self.modes = ALL_MODES
        self.tasks = ALL_TASKS
    
    def load_results(self, model_name: str) -> List[dict]:
        """Load extracted results for a model"""
        results_dir = RESULTS_DIR / model_name
        extracted_file = results_dir / "extracted.jsonl"
        
        if not extracted_file.exists():
            raise FileNotFoundError(f"Extracted results not found: {extracted_file}")
        
        results = []
        try:
            with open(extracted_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            result = json.loads(line.strip())
                            results.append(result)
                        except json.JSONDecodeError as e:
                            print(f"⚠️  JSON decode error on line {line_num}: {e}")
                            continue
        except Exception as e:
            raise Exception(f"Error loading results: {e}")
        
        return results
    
    def compute_metrics(self, results: List[dict]) -> Dict:
        """Compute comprehensive metrics from results"""
        if not results:
            return {}
        
        # Basic statistics
        total_samples = len(results)
        correct_samples = sum(1 for r in results if r.get("correct", False))
        overall_accuracy = correct_samples / total_samples if total_samples > 0 else 0
        
        # Group results for detailed analysis
        by_mode = defaultdict(list)
        by_task = defaultdict(list)
        by_domain = defaultdict(list)
        
        # Group by task and index for agreement analysis
        by_task_index = defaultdict(dict)
        
        for result in results:
            mode = result.get("mode", "unknown")
            task = result.get("task_name", "unknown")
            index = result.get("index", -1)
            domain = DOMAIN_MAPPING.get(task, "unknown")
            
            by_mode[mode].append(result)
            by_task[task].append(result)
            by_domain[domain].append(result)
            
            # Store by task_index for agreement analysis
            key = f"{task}_{index}"
            by_task_index[key][mode] = result
        
        # Mode-wise metrics
        mode_metrics = {}
        for mode, mode_results in by_mode.items():
            correct = sum(1 for r in mode_results if r.get("correct", False))
            total = len(mode_results)
            accuracy = correct / total if total > 0 else 0
            
            mode_metrics[mode] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total
            }
        
        # Task-wise metrics
        task_metrics = {}
        for task, task_results in by_task.items():
            correct = sum(1 for r in task_results if r.get("correct", False))
            total = len(task_results)
            accuracy = correct / total if total > 0 else 0
            
            # Mode breakdown for this task
            task_mode_breakdown = {}
            task_by_mode = defaultdict(list)
            for r in task_results:
                task_by_mode[r.get("mode", "unknown")].append(r)
            
            for mode, mode_results in task_by_mode.items():
                mode_correct = sum(1 for r in mode_results if r.get("correct", False))
                mode_total = len(mode_results)
                mode_acc = mode_correct / mode_total if mode_total > 0 else 0
                task_mode_breakdown[mode] = {
                    "accuracy": mode_acc,
                    "correct": mode_correct,
                    "total": mode_total
                }
            
            task_metrics[task] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "by_mode": task_mode_breakdown
            }
        
        # Domain-wise metrics
        domain_metrics = {}
        for domain, domain_results in by_domain.items():
            correct = sum(1 for r in domain_results if r.get("correct", False))
            total = len(domain_results)
            accuracy = correct / total if total > 0 else 0
            
            # Mode breakdown for this domain
            domain_mode_breakdown = {}
            domain_by_mode = defaultdict(list)
            for r in domain_results:
                domain_by_mode[r.get("mode", "unknown")].append(r)
            
            for mode, mode_results in domain_by_mode.items():
                mode_correct = sum(1 for r in mode_results if r.get("correct", False))
                mode_total = len(mode_results)
                mode_acc = mode_correct / mode_total if mode_total > 0 else 0
                domain_mode_breakdown[mode] = {
                    "accuracy": mode_acc,
                    "correct": mode_correct,
                    "total": mode_total
                }
            
            domain_metrics[domain] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "by_mode": domain_mode_breakdown
            }
        
        # Extraction method analysis
        extraction_stats = defaultdict(int)
        for result in results:
            method = result.get("extraction_method", "unknown")
            extraction_stats[method] += 1
        
        # Agreement analysis - cross-modal prediction agreement
        agreement_metrics = self.compute_agreement_metrics(by_task_index)
        
        # Removed latency statistics
        
        # Compile comprehensive metrics
        metrics = {
            "model": results[0].get("model", "unknown") if results else "unknown",
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "overall_accuracy": overall_accuracy,
            "by_mode": dict(mode_metrics),
            "by_task": dict(task_metrics),
            "by_domain": dict(domain_metrics),
            "extraction_stats": dict(extraction_stats),
            "agreement_metrics": agreement_metrics
        }
        
        return metrics
    
    def compute_agreement_metrics(self, by_task_index: Dict) -> Dict:
        """Compute cross-modal agreement metrics"""
        agreement_stats = {
            "l_v_agreement": {"total": 0, "agreed": 0, "percentage": 0.0},
            "l_vl_agreement": {"total": 0, "agreed": 0, "percentage": 0.0},
            "v_vl_agreement": {"total": 0, "agreed": 0, "percentage": 0.0},
            "all_three_agreement": {"total": 0, "agreed": 0, "percentage": 0.0},
            "by_task": {},
            "by_domain": {}
        }
        
        # Initialize domain stats
        domain_agreement = {}
        for domain in ["chess", "chemistry", "music", "graph"]:
            domain_agreement[domain] = {
                "l_v_agreement": {"total": 0, "agreed": 0, "percentage": 0.0},
                "l_vl_agreement": {"total": 0, "agreed": 0, "percentage": 0.0},
                "v_vl_agreement": {"total": 0, "agreed": 0, "percentage": 0.0},
                "all_three_agreement": {"total": 0, "agreed": 0, "percentage": 0.0}
            }
        
        # Initialize task stats
        task_agreement = {}
        for task in self.tasks:
            task_agreement[task] = {
                "l_v_agreement": {"total": 0, "agreed": 0, "percentage": 0.0},
                "l_vl_agreement": {"total": 0, "agreed": 0, "percentage": 0.0},
                "v_vl_agreement": {"total": 0, "agreed": 0, "percentage": 0.0},
                "all_three_agreement": {"total": 0, "agreed": 0, "percentage": 0.0}
            }
        
        # Process each question (task_index combination)
        for key, modes_data in by_task_index.items():
            task_name = "_".join(key.split("_")[:-1])  # Remove index from key
            domain = DOMAIN_MAPPING.get(task_name, "unknown")
            
            if domain == "unknown" or task_name not in task_agreement:
                continue
            
            # Get predictions for each mode (treat Z as disagreement)
            l_pred = modes_data.get("l", {}).get("final_answer", "Z")
            v_pred = modes_data.get("v", {}).get("final_answer", "Z")
            vl_pred = modes_data.get("vl", {}).get("final_answer", "Z")
            
            # Check pairwise agreements
            if "l" in modes_data and "v" in modes_data:
                agreement_stats["l_v_agreement"]["total"] += 1
                task_agreement[task_name]["l_v_agreement"]["total"] += 1
                domain_agreement[domain]["l_v_agreement"]["total"] += 1
                
                if l_pred == v_pred and l_pred != "Z":
                    agreement_stats["l_v_agreement"]["agreed"] += 1
                    task_agreement[task_name]["l_v_agreement"]["agreed"] += 1
                    domain_agreement[domain]["l_v_agreement"]["agreed"] += 1
            
            if "l" in modes_data and "vl" in modes_data:
                agreement_stats["l_vl_agreement"]["total"] += 1
                task_agreement[task_name]["l_vl_agreement"]["total"] += 1
                domain_agreement[domain]["l_vl_agreement"]["total"] += 1
                
                if l_pred == vl_pred and l_pred != "Z":
                    agreement_stats["l_vl_agreement"]["agreed"] += 1
                    task_agreement[task_name]["l_vl_agreement"]["agreed"] += 1
                    domain_agreement[domain]["l_vl_agreement"]["agreed"] += 1
            
            if "v" in modes_data and "vl" in modes_data:
                agreement_stats["v_vl_agreement"]["total"] += 1
                task_agreement[task_name]["v_vl_agreement"]["total"] += 1
                domain_agreement[domain]["v_vl_agreement"]["total"] += 1
                
                if v_pred == vl_pred and v_pred != "Z":
                    agreement_stats["v_vl_agreement"]["agreed"] += 1
                    task_agreement[task_name]["v_vl_agreement"]["agreed"] += 1
                    domain_agreement[domain]["v_vl_agreement"]["agreed"] += 1
            
            # Check three-way agreement
            if "l" in modes_data and "v" in modes_data and "vl" in modes_data:
                agreement_stats["all_three_agreement"]["total"] += 1
                task_agreement[task_name]["all_three_agreement"]["total"] += 1
                domain_agreement[domain]["all_three_agreement"]["total"] += 1
                
                if l_pred == v_pred == vl_pred and l_pred != "Z":
                    agreement_stats["all_three_agreement"]["agreed"] += 1
                    task_agreement[task_name]["all_three_agreement"]["agreed"] += 1
                    domain_agreement[domain]["all_three_agreement"]["agreed"] += 1
        
        # Calculate percentages for overall stats
        for agreement_type in ["l_v_agreement", "l_vl_agreement", "v_vl_agreement", "all_three_agreement"]:
            total = agreement_stats[agreement_type]["total"]
            agreed = agreement_stats[agreement_type]["agreed"]
            agreement_stats[agreement_type]["percentage"] = (agreed / total * 100) if total > 0 else 0.0
        
        # Calculate percentages for task stats
        for task_name in task_agreement:
            for agreement_type in ["l_v_agreement", "l_vl_agreement", "v_vl_agreement", "all_three_agreement"]:
                total = task_agreement[task_name][agreement_type]["total"]
                agreed = task_agreement[task_name][agreement_type]["agreed"]
                task_agreement[task_name][agreement_type]["percentage"] = (agreed / total * 100) if total > 0 else 0.0
        
        # Calculate percentages for domain stats
        for domain in domain_agreement:
            for agreement_type in ["l_v_agreement", "l_vl_agreement", "v_vl_agreement", "all_three_agreement"]:
                total = domain_agreement[domain][agreement_type]["total"]
                agreed = domain_agreement[domain][agreement_type]["agreed"]
                domain_agreement[domain][agreement_type]["percentage"] = (agreed / total * 100) if total > 0 else 0.0
        
        agreement_stats["by_task"] = task_agreement
        agreement_stats["by_domain"] = domain_agreement
        
        return agreement_stats
    
    def save_metrics(self, metrics: Dict, model_name: str):
        """Save metrics to JSON file"""
        results_dir = RESULTS_DIR / model_name
        metrics_file = results_dir / "metrics.json"
        
        try:
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Metrics saved: {metrics_file}")
            
        except Exception as e:
            print(f"❌ Error saving metrics: {e}")
    
    def print_summary(self, metrics: Dict):
        """Print formatted metrics summary"""
        model = metrics.get("model", "unknown")
        total = metrics.get("total_samples", 0)
        correct = metrics.get("correct_samples", 0)
        accuracy = metrics.get("overall_accuracy", 0)
        
        print(f"\n📊 SEAM Benchmark Results - {model}")
        print(f"   Total samples: {total}")
        print(f"   Correct answers: {correct}")
        print(f"   Overall accuracy: {accuracy:.3f} ({100*accuracy:.1f}%)")
        
        # Mode breakdown
        print(f"\n🔄 By Mode:")
        by_mode = metrics.get("by_mode", {})
        for mode in ["l", "v", "vl"]:
            if mode in by_mode:
                mode_data = by_mode[mode]
                acc = mode_data.get("accuracy", 0)
                total_mode = mode_data.get("total", 0)
                correct_mode = mode_data.get("correct", 0)
                print(f"   {mode.upper()}: {acc:.3f} ({correct_mode}/{total_mode})")
        
        # Domain breakdown
        print(f"\n🏗️ By Domain:")
        by_domain = metrics.get("by_domain", {})
        for domain in ["chess", "chemistry", "music", "graph"]:
            if domain in by_domain:
                domain_data = by_domain[domain]
                acc = domain_data.get("accuracy", 0)
                total_domain = domain_data.get("total", 0)
                correct_domain = domain_data.get("correct", 0)
                print(f"   {domain.capitalize()}: {acc:.3f} ({correct_domain}/{total_domain})")
        
        # Extraction method stats
        extraction_stats = metrics.get("extraction_stats", {})
        if extraction_stats:
            print(f"\n🔍 Extraction Methods:")
            for method, count in extraction_stats.items():
                percentage = (count / total) * 100 if total > 0 else 0
                print(f"   {method}: {count} ({percentage:.1f}%)")
        
        # Agreement metrics
        agreement_metrics = metrics.get("agreement_metrics", {})
        if agreement_metrics:
            print(f"\n🤝 Cross-Modal Agreement:")
            print(f"   L↔V:  {agreement_metrics['l_v_agreement']['agreed']}/{agreement_metrics['l_v_agreement']['total']} ({agreement_metrics['l_v_agreement']['percentage']:.1f}%)")
            print(f"   L↔VL: {agreement_metrics['l_vl_agreement']['agreed']}/{agreement_metrics['l_vl_agreement']['total']} ({agreement_metrics['l_vl_agreement']['percentage']:.1f}%)")
            print(f"   V↔VL: {agreement_metrics['v_vl_agreement']['agreed']}/{agreement_metrics['v_vl_agreement']['total']} ({agreement_metrics['v_vl_agreement']['percentage']:.1f}%)")
            print(f"   All:  {agreement_metrics['all_three_agreement']['agreed']}/{agreement_metrics['all_three_agreement']['total']} ({agreement_metrics['all_three_agreement']['percentage']:.1f}%)")
        
        # Performance stats removed

def find_all_models() -> List[str]:
    """Find all model directories with extracted results"""
    results_dir = RESULTS_DIR
    if not results_dir.exists():
        return []
    
    models = []
    for item in results_dir.iterdir():
        if item.is_dir() and (item / "extracted.jsonl").exists():
            models.append(item.name)
    
    return sorted(models)

def generate_comparison_plots(models: List[str], output_dir: Path):
    """Generate comparison plots for multiple models"""
    if not PLOTTING_AVAILABLE:
        print("⚠️  Matplotlib/seaborn not available. Skipping plots.")
        return
    
    print(f"\n📈 Generating comparison plots...")
    
    # Load metrics for all models
    all_metrics = {}
    for model in models:
        try:
            metrics_file = RESULTS_DIR / model / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    all_metrics[model] = json.load(f)
        except Exception as e:
            print(f"⚠️  Could not load metrics for {model}: {e}")
    
    if len(all_metrics) < 2:
        print("⚠️  Need at least 2 models for comparison plots")
        return
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Overall accuracy comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('SEAM Benchmark Model Comparison', fontsize=16, fontweight='bold')
    
    # Overall accuracy bar plot
    ax1 = axes[0]
    model_names = list(all_metrics.keys())
    overall_accs = [all_metrics[m].get("overall_accuracy", 0) for m in model_names]
    
    bars = ax1.bar(range(len(model_names)), overall_accs)
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Overall Accuracy')
    ax1.set_title('Overall Accuracy by Model')
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels([m.replace('-', '\\n') for m in model_names], rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, overall_accs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 2. Mode comparison
    ax2 = axes[1]
    mode_data = defaultdict(list)
    
    for model in model_names:
        by_mode = all_metrics[model].get("by_mode", {})
        for mode in ["l", "v", "vl"]:
            mode_acc = by_mode.get(mode, {}).get("accuracy", 0)
            mode_data[mode].append(mode_acc)
    
    x = np.arange(len(model_names))
    width = 0.25
    
    for i, mode in enumerate(["l", "v", "vl"]):
        offset = (i - 1) * width
        ax2.bar(x + offset, mode_data[mode], width, label=f'{mode.upper()} Mode')
    
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy by Mode')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace('-', '\\n') for m in model_names], rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    # 3. Domain comparison
    ax3 = axes[2]
    domain_data = defaultdict(list)
    
    for model in model_names:
        by_domain = all_metrics[model].get("by_domain", {})
        for domain in ["chess", "chemistry", "music", "graph"]:
            domain_acc = by_domain.get(domain, {}).get("accuracy", 0)
            domain_data[domain].append(domain_acc)
    
    x = np.arange(len(model_names))
    width = 0.2
    
    for i, domain in enumerate(["chess", "chemistry", "music", "graph"]):
        offset = (i - 1.5) * width
        ax3.bar(x + offset, domain_data[domain], width, label=domain.capitalize())
    
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Accuracy by Domain')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.replace('-', '\\n') for m in model_names], rotation=45, ha='right')
    ax3.legend()
    ax3.set_ylim(0, 1)
    
    # Removed latency comparison plot
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / "comparison_plots.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Comparison plots saved: {plot_file}")

def main():
    parser = argparse.ArgumentParser(
        description='SEAM Benchmark - Unified Metrics Computation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute metrics for specific model
  python 03_metric.py --model qwen-qwen2.5-vl-7b-instruct
  
  # Compute metrics for all models
  python 03_metric.py --all
  
  # Generate comparison plots
  python 03_metric.py --compare --models qwen-qwen2.5-vl-7b-instruct,gpt-4o-mini
  
  # List available models
  python 03_metric.py --list
        """
    )
    
    # Model selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', help='Specific model name to compute metrics')
    group.add_argument('--all', action='store_true', help='Compute metrics for all models')
    group.add_argument('--compare', action='store_true', help='Generate comparison plots')
    group.add_argument('--list', action='store_true', help='List available models')
    
    # Comparison options
    parser.add_argument('--models', help='Comma-separated model names for comparison')
    parser.add_argument('--output-dir', default=str(RESULTS_DIR),
                       help='Output directory for plots (default: results/)')
    
    args = parser.parse_args()
    
    try:
        calculator = MetricsCalculator()
        
        if args.list:
            # List available models
            models = find_all_models()
            if not models:
                print("No models with extracted results found in results/")
                return 0
            
            print(f"Available models with extracted results ({len(models)}):")
            for model in models:
                results_dir = RESULTS_DIR / model
                metrics_file = results_dir / "metrics.json"
                
                status = "✅ computed" if metrics_file.exists() else "⏳ pending"
                
                # Count samples
                sample_count = 0
                extracted_file = results_dir / "extracted.jsonl"
                if extracted_file.exists():
                    try:
                        with open(extracted_file, 'r') as f:
                            sample_count = sum(1 for line in f if line.strip())
                    except:
                        pass
                
                print(f"  • {model} ({sample_count} samples) - {status}")
            
            return 0
        
        elif args.compare:
            # Generate comparison plots
            if args.models:
                models = [m.strip() for m in args.models.split(',')]
            else:
                models = find_all_models()
            
            if len(models) < 2:
                print("❌ Need at least 2 models for comparison")
                return 1
            
            print(f"📊 Generating comparison for {len(models)} models:")
            for model in models:
                print(f"   • {model}")
            
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            
            generate_comparison_plots(models, output_dir)
            
            print(f"\n🎉 Comparison plots generated!")
            
        elif args.all:
            # Process all models
            models = find_all_models()
            if not models:
                print("No models with extracted results found in results/")
                return 1
            
            print(f"📊 Computing metrics for {len(models)} models...")
            
            for model in models:
                print(f"\n🔄 Processing: {model}")
                try:
                    results = calculator.load_results(model)
                    metrics = calculator.compute_metrics(results)
                    calculator.save_metrics(metrics, model)
                    calculator.print_summary(metrics)
                except Exception as e:
                    print(f"❌ Error processing {model}: {e}")
            
            print(f"\n🎉 Batch metrics computation completed!")
            
        else:
            # Process specific model
            print(f"📊 Computing metrics for: {args.model}")
            
            results = calculator.load_results(args.model)
            metrics = calculator.compute_metrics(results)
            calculator.save_metrics(metrics, args.model)
            calculator.print_summary(metrics)
            
            print(f"\n🎉 Metrics computation completed!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)