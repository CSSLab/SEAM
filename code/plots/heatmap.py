import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import numpy as np
import json
from pathlib import Path

# Set the style for publication-quality figures - EXACT same as original
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Increase font sizes for readability in publication
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10



def parse_heatmap_domain_data(text, domain_name):
    """Parse domain data from text"""
    data = []
    lines = text.strip().split('\n')
    
    for line in lines:
        if 'Average Accuracy' not in line:
            continue
        parts = line.split(', Average Accuracy ')
        model = parts[0].replace('Model: ', '')
        metric_type = parts[1].split(': ')[0]
        value = float(parts[1].split(': ')[1])
        data.append({'Model': model, 'Metric': metric_type, 'Accuracy': value, 'Domain': domain_name})
    
    return data


def clean_model_name(model_name):
    """Clean model name for display"""
    # Add hyphens to connect model names for better display
    if 'GPT-5' in model_name and 'mini' in model_name:
        return 'GPT-5-mini'
    elif 'GPT-5' in model_name and 'nano' in model_name:
        return 'GPT-5-nano'
    elif 'GPT-5' == model_name:
        return 'GPT-5'
    elif 'GPT-4o-mini' in model_name:
        return 'GPT-4o-mini'
    elif 'GPT-4o' in model_name:
        return 'GPT-4o'
    elif 'Claude-4.1 Opus' in model_name:
        return 'Claude-4.1-Opus'
    elif 'Claude-4 Sonnet' in model_name:
        return 'Claude-4-Sonnet'
    elif 'Claude-3.7 Sonnet' in model_name:
        return 'Claude-3.7-Sonnet'
    elif 'Claude-3.5 Sonnet' in model_name:
        return 'Claude-3.5-Sonnet'
    elif 'Claude-3.5 Haiku' in model_name:
        return 'Claude-3.5-Haiku'
    elif 'Qwen2.5-VL-72B-Instruct' in model_name:
        return 'Qwen2.5-VL-72B-Instruct'
    elif 'Qwen2.5-VL-7B-Instruct' in model_name:
        return 'Qwen2.5-VL-7B-Instruct'
    elif 'Qwen2.5-Omni-7B' in model_name:
        return 'Qwen2.5-Omni-7B'
    elif 'InternVL3-78B' in model_name:
        return 'InternVL3-78B'
    elif 'InternVL3-8B' in model_name:
        return 'InternVL3-8B'
    elif 'InternVL-2.5-78B' in model_name:
        return 'InternVL-2.5-78B'
    elif 'InternVL-2.5-8B' in model_name:
        return 'InternVL-2.5-8B'
    elif 'Llama-3.2-90B-Vision-Instruct' in model_name:
        return 'Llama-3.2-90B-Vision-Instruct'
    elif 'Llama-3.2-11B-Vision-Instruct' in model_name:
        return 'Llama-3.2-11B-Vision-Instruct'
    elif 'gemma-3-27b-it' in model_name:
        return 'gemma-3-27b-it'
    elif 'gemma-3-12b-it' in model_name:
        return 'gemma-3-12b-it'
    else:
        return model_name


def load_new_model_data(results_dir='results'):
    """Load data from results directory"""
    results_path = Path(results_dir)
    model_data = {}
    
    domain_tasks = {
        'chess': ['fork', 'legal', 'puzzle', 'eval'],
        'chem': ['carbon', 'hydrogen', 'weight', 'caption'],
        'music': ['notes', 'measures', 'forms', 'rhythm'],
        'graph': ['path_counting', 'path_existence', 'shortest_path', 'bfs_traversal']
    }
    
    for model_dir in results_path.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            results_file = model_dir / 'results.jsonl'
            
            if results_file.exists():
                results = []
                with open(results_file, 'r') as f:
                    for line in f:
                        results.append(json.loads(line))
                
                # Store raw results for agreement calculation
                model_data[model_name] = {
                    'raw_results': results,
                    'domain_mode_stats': {}
                }
                
                # Compute accuracy by domain and mode
                for domain, tasks in domain_tasks.items():
                    model_data[model_name]['domain_mode_stats'][domain] = {}
                    for mode in ['l', 'v', 'vl']:
                        filtered = [r for r in results if r['task_name'] in tasks and r['mode'] == mode]
                        if filtered:
                            accuracy = sum(1 for r in filtered if r['correct']) / len(filtered)
                            model_data[model_name]['domain_mode_stats'][domain][mode] = accuracy
                        else:
                            model_data[model_name]['domain_mode_stats'][domain][mode] = 0.0
                            
    return model_data


def compute_agreement_and_accuracy(model_data):
    """Compute agreement (L-V) and average accuracy for new models"""
    agreement_data = []
    
    for model_name, data in model_data.items():
        # Skip InternVL3-14B
        if 'InternVL3-14B' in model_name:
            continue
            
        raw_results = data['raw_results']
        domain_stats = data['domain_mode_stats']
        
        # Sort results by index to ensure proper matching
        df = pd.DataFrame(raw_results)
        df = df.sort_values('index')
        
        # Calculate agreement as exact match between L and V predictions
        # We need to match questions across modes based on their position
        l_results = df[df['mode'] == 'l'].reset_index(drop=True)
        v_results = df[df['mode'] == 'v'].reset_index(drop=True)
        
        # Calculate agreement
        agreement_count = 0
        total_count = 0
        
        # Match questions by their position after sorting
        for i in range(min(len(l_results), len(v_results))):
            l_pred = l_results.iloc[i].get('final_answer', l_results.iloc[i].get('output', 'Z'))
            v_pred = v_results.iloc[i].get('final_answer', v_results.iloc[i].get('output', 'Z'))
            
            # Count exact matches, excluding invalid responses
            if l_pred == v_pred and l_pred != 'Z':
                agreement_count += 1
            if l_pred != 'Z' or v_pred != 'Z':  # Count valid pairs
                total_count += 1
        
        if total_count > 0:
            agreement = agreement_count / total_count
        else:
            agreement = 0.0
        
        # Calculate average accuracy across all modes and domains
        all_accuracies = []
        for domain, modes in domain_stats.items():
            for mode_acc in modes.values():
                all_accuracies.append(mode_acc)
        
        avg_accuracy = np.mean(all_accuracies) if all_accuracies else 0.0
        
        # Determine series based on model name
        clean_name = clean_model_name(model_name)
        if 'InternVL3' in clean_name:
            series = 'InternVL3'
        elif 'Qwen2.5-Omni' in clean_name:
            series = 'Qwen-Omni'
        else:
            series = 'Other'
        
        agreement_data.append({
            'Model': clean_name,
            'Agreement_L_V': agreement,
            'Avg_Accuracy': avg_accuracy,
            'Series': series
        })
    
    return pd.DataFrame(agreement_data)


# Removed old agreement plot function - not needed for this update


def parse_comment_data():
    """Parse the hardcoded data from the comments at the top of this file"""
    data = []
    
    # Chess Domain data
    chess_data = {
        'GPT-5': {'Language': 0.710, 'Vision': 0.746, 'Vision-Language': 0.734},
        'GPT-5-mini': {'Language': 0.776, 'Vision': 0.786, 'Vision-Language': 0.759},
        'GPT-5-nano': {'Language': 0.743, 'Vision': 0.659, 'Vision-Language': 0.731},
        'GPT-4o': {'Language': 0.624, 'Vision': 0.644, 'Vision-Language': 0.624},
        'GPT-4o-mini': {'Language': 0.610, 'Vision': 0.646, 'Vision-Language': 0.634},
        'Claude-4.1-Opus': {'Language': 0.806, 'Vision': 0.718, 'Vision-Language': 0.794},
        'Claude-4-Sonnet': {'Language': 0.799, 'Vision': 0.734, 'Vision-Language': 0.784},
        'Claude-3.7-Sonnet': {'Language': 0.706, 'Vision': 0.656, 'Vision-Language': 0.675},
        'Claude-3.5-Sonnet': {'Language': 0.652, 'Vision': 0.615, 'Vision-Language': 0.651},
        'Claude-3.5-Haiku': {'Language': 0.623, 'Vision': 0.549, 'Vision-Language': 0.635},
        'Qwen2.5-VL-72B-Instruct': {'Language': 0.542, 'Vision': 0.586, 'Vision-Language': 0.571},
        'Qwen2.5-VL-7B-Instruct': {'Language': 0.295, 'Vision': 0.389, 'Vision-Language': 0.386},
        'Qwen2.5-Omni-7B': {'Language': 0.501, 'Vision': 0.512, 'Vision-Language': 0.448},
        'InternVL3-78B': {'Language': 0.549, 'Vision': 0.549, 'Vision-Language': 0.560},
        'InternVL3-8B': {'Language': 0.420, 'Vision': 0.431, 'Vision-Language': 0.454},
        'InternVL-2.5-78B': {'Language': 0.552, 'Vision': 0.631, 'Vision-Language': 0.635},
        'InternVL-2.5-8B': {'Language': 0.501, 'Vision': 0.466, 'Vision-Language': 0.450},
        'Llama-3.2-90B-Vision-Instruct': {'Language': 0.495, 'Vision': 0.538, 'Vision-Language': 0.535},
        'Llama-3.2-11B-Vision-Instruct': {'Language': 0.312, 'Vision': 0.446, 'Vision-Language': 0.415},
        'gemma-3-27b-it': {'Language': 0.522, 'Vision': 0.545, 'Vision-Language': 0.517},
        'gemma-3-12b-it': {'Language': 0.454, 'Vision': 0.490, 'Vision-Language': 0.476}
    }
    
    # Chemistry Domain data
    chemistry_data = {
        'GPT-5': {'Language': 0.910, 'Vision': 0.758, 'Vision-Language': 0.933},
        'GPT-5-mini': {'Language': 0.846, 'Vision': 0.779, 'Vision-Language': 0.834},
        'GPT-5-nano': {'Language': 0.739, 'Vision': 0.528, 'Vision-Language': 0.744},
        'GPT-4o': {'Language': 0.650, 'Vision': 0.573, 'Vision-Language': 0.653},
        'GPT-4o-mini': {'Language': 0.529, 'Vision': 0.409, 'Vision-Language': 0.489},
        'Claude-4.1-Opus': {'Language': 0.945, 'Vision': 0.851, 'Vision-Language': 0.946},
        'Claude-4-Sonnet': {'Language': 0.891, 'Vision': 0.704, 'Vision-Language': 0.904},
        'Claude-3.7-Sonnet': {'Language': 0.888, 'Vision': 0.803, 'Vision-Language': 0.870},
        'Claude-3.5-Sonnet': {'Language': 0.836, 'Vision': 0.813, 'Vision-Language': 0.688},
        'Claude-3.5-Haiku': {'Language': 0.574, 'Vision': 0.529, 'Vision-Language': 0.589},
        'Qwen2.5-VL-72B-Instruct': {'Language': 0.575, 'Vision': 0.559, 'Vision-Language': 0.609},
        'Qwen2.5-VL-7B-Instruct': {'Language': 0.296, 'Vision': 0.443, 'Vision-Language': 0.410},
        'Qwen2.5-Omni-7B': {'Language': 0.310, 'Vision': 0.307, 'Vision-Language': 0.338},
        'InternVL3-78B': {'Language': 0.468, 'Vision': 0.446, 'Vision-Language': 0.505},
        'InternVL3-8B': {'Language': 0.372, 'Vision': 0.289, 'Vision-Language': 0.364},
        'InternVL-2.5-78B': {'Language': 0.448, 'Vision': 0.404, 'Vision-Language': 0.460},
        'InternVL-2.5-8B': {'Language': 0.219, 'Vision': 0.328, 'Vision-Language': 0.300},
        'Llama-3.2-90B-Vision-Instruct': {'Language': 0.501, 'Vision': 0.363, 'Vision-Language': 0.551},
        'Llama-3.2-11B-Vision-Instruct': {'Language': 0.304, 'Vision': 0.294, 'Vision-Language': 0.340},
        'gemma-3-27b-it': {'Language': 0.529, 'Vision': 0.499, 'Vision-Language': 0.515},
        'gemma-3-12b-it': {'Language': 0.463, 'Vision': 0.446, 'Vision-Language': 0.509}
    }
    
    # Music Domain data
    music_data = {
        'GPT-5': {'Language': 0.806, 'Vision': 0.343, 'Vision-Language': 0.764},
        'GPT-5-mini': {'Language': 0.736, 'Vision': 0.348, 'Vision-Language': 0.730},
        'GPT-5-nano': {'Language': 0.548, 'Vision': 0.328, 'Vision-Language': 0.549},
        'GPT-4o': {'Language': 0.540, 'Vision': 0.328, 'Vision-Language': 0.495},
        'GPT-4o-mini': {'Language': 0.348, 'Vision': 0.241, 'Vision-Language': 0.307},
        'Claude-4.1-Opus': {'Language': 0.581, 'Vision': 0.348, 'Vision-Language': 0.580},
        'Claude-4-Sonnet': {'Language': 0.569, 'Vision': 0.328, 'Vision-Language': 0.555},
        'Claude-3.7-Sonnet': {'Language': 0.509, 'Vision': 0.409, 'Vision-Language': 0.431},
        'Claude-3.5-Sonnet': {'Language': 0.431, 'Vision': 0.365, 'Vision-Language': 0.344},
        'Claude-3.5-Haiku': {'Language': 0.361, 'Vision': 0.301, 'Vision-Language': 0.290},
        'Qwen2.5-VL-72B-Instruct': {'Language': 0.425, 'Vision': 0.341, 'Vision-Language': 0.373},
        'Qwen2.5-VL-7B-Instruct': {'Language': 0.282, 'Vision': 0.260, 'Vision-Language': 0.280},
        'Qwen2.5-Omni-7B': {'Language': 0.289, 'Vision': 0.279, 'Vision-Language': 0.306},
        'InternVL3-78B': {'Language': 0.411, 'Vision': 0.285, 'Vision-Language': 0.361},
        'InternVL3-8B': {'Language': 0.302, 'Vision': 0.289, 'Vision-Language': 0.278},
        'InternVL-2.5-78B': {'Language': 0.319, 'Vision': 0.236, 'Vision-Language': 0.283},
        'InternVL-2.5-8B': {'Language': 0.245, 'Vision': 0.220, 'Vision-Language': 0.221},
        'Llama-3.2-90B-Vision-Instruct': {'Language': 0.331, 'Vision': 0.269, 'Vision-Language': 0.276},
        'Llama-3.2-11B-Vision-Instruct': {'Language': 0.264, 'Vision': 0.245, 'Vision-Language': 0.253},
        'gemma-3-27b-it': {'Language': 0.430, 'Vision': 0.330, 'Vision-Language': 0.354},
        'gemma-3-12b-it': {'Language': 0.395, 'Vision': 0.280, 'Vision-Language': 0.316}
    }
    
    # Graph Domain data
    graph_data = {
        'GPT-5': {'Language': 0.791, 'Vision': 0.684, 'Vision-Language': 0.999},
        'GPT-5-mini': {'Language': 0.788, 'Vision': 0.699, 'Vision-Language': 0.996},
        'GPT-5-nano': {'Language': 0.768, 'Vision': 0.525, 'Vision-Language': 0.990},
        'GPT-4o': {'Language': 0.725, 'Vision': 0.382, 'Vision-Language': 0.738},
        'GPT-4o-mini': {'Language': 0.735, 'Vision': 0.347, 'Vision-Language': 0.688},
        'Claude-4.1-Opus': {'Language': 0.974, 'Vision': 0.394, 'Vision-Language': 0.936},
        'Claude-4-Sonnet': {'Language': 0.974, 'Vision': 0.414, 'Vision-Language': 0.971},
        'Claude-3.7-Sonnet': {'Language': 0.868, 'Vision': 0.496, 'Vision-Language': 0.739},
        'Claude-3.5-Sonnet': {'Language': 0.739, 'Vision': 0.446, 'Vision-Language': 0.473},
        'Claude-3.5-Haiku': {'Language': 0.561, 'Vision': 0.353, 'Vision-Language': 0.471},
        'Qwen2.5-VL-72B-Instruct': {'Language': 0.647, 'Vision': 0.414, 'Vision-Language': 0.524},
        'Qwen2.5-VL-7B-Instruct': {'Language': 0.340, 'Vision': 0.307, 'Vision-Language': 0.361},
        'Qwen2.5-Omni-7B': {'Language': 0.351, 'Vision': 0.318, 'Vision-Language': 0.364},
        'InternVL3-78B': {'Language': 0.574, 'Vision': 0.430, 'Vision-Language': 0.502},
        'InternVL3-8B': {'Language': 0.434, 'Vision': 0.419, 'Vision-Language': 0.448},
        'InternVL-2.5-78B': {'Language': 0.471, 'Vision': 0.386, 'Vision-Language': 0.460},
        'InternVL-2.5-8B': {'Language': 0.331, 'Vision': 0.336, 'Vision-Language': 0.365},
        'Llama-3.2-90B-Vision-Instruct': {'Language': 0.409, 'Vision': 0.367, 'Vision-Language': 0.393},
        'Llama-3.2-11B-Vision-Instruct': {'Language': 0.275, 'Vision': 0.335, 'Vision-Language': 0.283},
        'gemma-3-27b-it': {'Language': 0.581, 'Vision': 0.339, 'Vision-Language': 0.414},
        'gemma-3-12b-it': {'Language': 0.521, 'Vision': 0.386, 'Vision-Language': 0.416}
    }
    
    # Convert to the expected format
    domain_mappings = {
        'chess': chess_data,
        'chem': chemistry_data,
        'music': music_data,
        'graph': graph_data
    }
    
    for domain, models in domain_mappings.items():
        for model, modes in models.items():
            for mode, accuracy in modes.items():
                data.append({
                    'Model': model,
                    'Clean Model': clean_model_name(model),
                    'Domain': domain,
                    'Metric': mode.replace('-', ' '),  # Convert "Vision-Language" to "Vision Language"
                    'Accuracy': accuracy
                })
    
    return pd.DataFrame(data)

def create_combined_heatmap(output_file='combined_heatmap.pdf'):
    """Create heatmap using the new data from comments"""
    # Get data from hardcoded comment data
    combined_df = parse_comment_data()
    
    # Create figure with GridSpec - EXACT original size
    fig = plt.figure(figsize=(11, 15))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Custom colormap
    colors = [(1, 1, 1), (0.8, 0.8, 1), (0, 0, 0.8)]
    custom_cmap = LinearSegmentedColormap.from_list('white_to_blue', colors, N=100)
    vmin = 0.2
    vmax = 0.9
    
    # Domain grid order
    domains_grid = {(0, 0): 'chess', (0, 1): 'chem', (1, 0): 'music', (1, 1): 'graph'}
    domain_names = {'chess': 'Chess', 'chem': 'Chemistry', 'music': 'Music', 'graph': 'Graph'}
    # Column mapping not needed anymore since data already has correct column names
    
    # Define model order - updated to include new models with hyphens
    custom_model_order = [
        'GPT-5', 'GPT-5-mini', 'GPT-5-nano', 'GPT-4o', 'GPT-4o-mini', 
        'Claude-4.1-Opus', 'Claude-4-Sonnet', 'Claude-3.7-Sonnet', 'Claude-3.5-Sonnet', 'Claude-3.5-Haiku',
        'Qwen2.5-VL-72B-Instruct', 'Qwen2.5-VL-7B-Instruct', 'Qwen2.5-Omni-7B',
        'InternVL3-78B', 'InternVL3-8B', 'InternVL-2.5-78B', 'InternVL-2.5-8B',
        'Llama-3.2-90B-Vision-Instruct', 'Llama-3.2-11B-Vision-Instruct', 'gemma-3-27b-it', 'gemma-3-12b-it'
    ]
    
    # Plot each domain
    for pos, domain in domains_grid.items():
        domain_df = combined_df[combined_df['Domain'] == domain]
        
        if domain_df.empty:
            continue
        
        pivot = pd.pivot_table(
            domain_df, values='Accuracy', index='Clean Model', columns='Metric', aggfunc='mean'
        )
        
        # Reorder columns to match expected format
        expected_cols = ['Language', 'Vision', 'Vision Language']
        available_cols = [col for col in expected_cols if col in pivot.columns]
        if available_cols:
            pivot = pivot[available_cols]
        
        # No need to rename columns as they are already in the correct format
        
        # Reorder rows by custom model order
        available_models = [model for model in custom_model_order if model in pivot.index]
        pivot = pivot.reindex(available_models)
        
        ax = fig.add_subplot(gs[pos])
        
        sns.heatmap(
            pivot, annot=True, fmt='.3f', cmap=custom_cmap, vmin=vmin, vmax=vmax,
            ax=ax, cbar=False, linewidths=0.5, annot_kws={"size": 10.5}
        )
        
        ax.set_title(domain_names[domain], fontsize=16, fontweight='bold')
        
        if pos[1] == 1:
            ax.set_yticklabels([])
            ax.set_ylabel('')
        else:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
            ax.set_ylabel('')
        
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
        ax.set_xlabel('')
    
    # Create a single colorbar
    cbar_ax = fig.add_axes([0.25, 0.09, 0.7, 0.015])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Accuracy', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.11, 1, 0.98])
    plt.subplots_adjust(bottom=0.16, hspace=0.2)
    
    # Save figure
    plt.savefig(output_file.replace('.pdf', '.png'), dpi=300)
    plt.close()
    
    print(f"Combined heatmap saved to {output_file}")


if __name__ == '__main__':
    # Generate heatmap plot using new data
    create_combined_heatmap('./heatmap.png')
    print("Combined heatmap generated successfully!")