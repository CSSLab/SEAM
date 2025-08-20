import matplotlib.pyplot as plt
import numpy as np

# Data extracted from updated results table
models_data = {
    # Model name: [Average Accuracy, L-V Agreement, Family] - flipped x and y
    # Proprietary Models
    'GPT-5': [0.765, 0.627, 'GPT'],
    'GPT-5-mini': [0.756, 0.630, 'GPT'],
    'GPT-5-nano': [0.654, 0.500, 'GPT'],
    'GPT-4o': [0.581, 0.503, 'GPT'],
    'GPT-4o-mini': [0.498, 0.480, 'GPT'],
    'Claude-4.1-Opus': [0.740, 0.575, 'Claude'],
    'Claude-4-Sonnet': [0.719, 0.569, 'Claude'],
    'Claude-3.7-Sonnet': [0.671, 0.594, 'Claude'],
    'Claude-3.5-Sonnet': [0.580, 0.537, 'Claude'],
    'Claude-3.5-Haiku': [0.486, 0.479, 'Claude'],
    # Open-Source Models
    'Qwen2.5-VL-72B-Instruct': [0.514, 0.447, 'Qwen'],
    'Qwen2.5-VL-7B-Instruct': [0.337, 0.347, 'Qwen'],
    'Qwen2.5-Omni-7B': [0.360, 0.353, 'Qwen'],
    'InternVL3-78B': [0.478, 0.447, 'InternVL3'],
    'InternVL3-8B': [0.375, 0.388, 'InternVL3'],
    'InternVL-2.5-78B': [0.440, 0.415, 'InternVL2.5'],
    'InternVL-2.5-8B': [0.332, 0.324, 'InternVL2.5'],
    'Llama-3.2-90B-Vision-Instruct': [0.419, 0.384, 'Llama'],
    'Llama-3.2-11B-Vision-Instruct': [0.314, 0.287, 'Llama'],
    'gemma-3-27b-it': [0.465, 0.447, 'Gemma'],
    'gemma-3-12b-it': [0.429, 0.419, 'Gemma']
}

# Random experiments data for calibration curve
random_experiments = {
    'target_acc': [0.000, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 
                   0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 1.000],
    'actual_acc': [0.000, 0.051, 0.100, 0.151, 0.199, 0.251, 0.300, 0.345, 0.403, 0.446,
                   0.495, 0.552, 0.604, 0.651, 0.702, 0.749, 0.800, 0.848, 0.900, 0.950, 1.000],
    'lv_agreement': [0.332, 0.303, 0.284, 0.259, 0.260, 0.253, 0.251, 0.261, 0.282, 0.298,
                     0.335, 0.382, 0.421, 0.459, 0.522, 0.585, 0.652, 0.722, 0.815, 0.903, 1.000]
}

# Define colors for each family
family_colors = {
    'GPT': '#1f77b4',      # blue
    'Claude': '#ff69b4',   # pink
    'Qwen': '#ff7f0e',     # orange
    'InternVL2.5': '#2ca02c', # green
    'InternVL3': '#8c564b', # brown
    'Llama': '#9467bd',    # purple
    'Gemma': '#17becf'     # light blue
}

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(24, 20))

# Group models by family for drawing connecting lines - flipped axes
family_models = {}
for model_name, (accuracy, agreement, family) in models_data.items():
    if family not in family_models:
        family_models[family] = []
    family_models[family].append((accuracy, agreement, model_name))

# Draw connecting lines within each family
for family, models in family_models.items():
    if len(models) > 1:
        # Sort models by accuracy score to connect them in order (now x-axis)
        models_sorted = sorted(models, key=lambda x: x[0])
        accuracies = [m[0] for m in models_sorted]
        agreements = [m[1] for m in models_sorted]
        ax.plot(accuracies, agreements, color=family_colors[family], alpha=0.7, linewidth=4)

# Extract model identifier from model name
def get_model_identifier(model_name):
    # GPT models
    if model_name == 'GPT-5':
        return 'GPT-5'
    elif model_name == 'GPT-5-mini':
        return 'GPT-5-mini'
    elif model_name == 'GPT-5-nano':
        return 'GPT-5-nano'
    elif model_name == 'GPT-4o':
        return 'GPT-4o'
    elif model_name == 'GPT-4o-mini':
        return 'GPT-4o-mini'
    # Claude models
    elif model_name == 'Claude-4.1-Opus':
        return 'Claude-4.1-Opus'
    elif model_name == 'Claude-4-Sonnet':
        return 'Claude-4-Sonnet'
    elif model_name == 'Claude-3.7-Sonnet':
        return 'Claude-3.7-Sonnet'
    elif model_name == 'Claude-3.5-Sonnet':
        return 'Claude-3.5-Sonnet'
    elif model_name == 'Claude-3.5-Haiku':
        return 'Claude-3.5-Haiku'
    # Qwen models
    elif model_name == 'Qwen2.5-VL-72B-Instruct':
        return 'Qwen2.5-VL-72B'
    elif model_name == 'Qwen2.5-VL-7B-Instruct':
        return 'Qwen2.5-VL-7B'
    elif model_name == 'Qwen2.5-Omni-7B':
        return 'Qwen2.5-Omni-7B'
    # InternVL models
    elif model_name == 'InternVL3-78B':
        return 'InternVL3-78B'
    elif model_name == 'InternVL3-8B':
        return 'InternVL3-8B'
    elif model_name == 'InternVL-2.5-78B':
        return 'InternVL2.5-78B'
    elif model_name == 'InternVL-2.5-8B':
        return 'InternVL2.5-8B'
    # Llama models
    elif model_name == 'Llama-3.2-90B-Vision-Instruct':
        return 'Llama3.2-90B'
    elif model_name == 'Llama-3.2-11B-Vision-Instruct':
        return 'Llama3.2-11B'
    # Gemma models
    elif model_name == 'gemma-3-27b-it':
        return 'Gemma3-27B'
    elif model_name == 'gemma-3-12b-it':
        return 'Gemma3-12B'
    else:
        return ''

# Plot theoretical baseline curve f(p) = p^2 + (1-p)^2/3
p_values = np.linspace(0.23, 0.84, 200)
baseline_values = p_values**2 + (1 - p_values)**2 / 3
ax.plot(p_values, baseline_values, color='#666666', linewidth=5, alpha=0.8, zorder=1)

# Plot each model with same marker style - flipped axes
for model_name, (accuracy, agreement, family) in models_data.items():
    color = family_colors[family]
    ax.scatter(accuracy, agreement, c=color, s=150, alpha=0.9, edgecolors='white', linewidth=1.5, zorder=3)
    
# Custom positioning for each model to avoid all overlaps
label_offsets = {
    # GPT models - high performers moved far with leader lines
    'GPT-5': (50, 40, 'right'),  # Far left and up with leader line
    'GPT-5-mini': (-10, 10, 'right'),  # Far left with leader line
    'GPT-5-nano': (0, 30, 'right'),  # Far left with leader line
    'GPT-4o': (-20, -30, 'left'),
    'GPT-4o-mini': (10, -10, 'left'),
    # Claude models - high performers moved far with leader lines
    'Claude-4.1-Opus': (240, -25, 'right'),  # Far left and down with leader line
    'Claude-4-Sonnet': (240, -70, 'right'),  # Far left and down with leader line
    'Claude-3.7-Sonnet': (-8, 8, 'right'),
    'Claude-3.5-Sonnet': (-15, 5, 'right'),
    'Claude-3.5-Haiku': (-15, 5, 'right'),
    # Qwen models
    'Qwen2.5-VL-72B-Instruct': (25, 0, 'left'),
    'Qwen2.5-VL-7B-Instruct': (-25, -10, 'right'),
    'Qwen2.5-Omni-7B': (-25, 8, 'right'),
    # InternVL models
    'InternVL3-78B': (-80, 40, 'right'),
    'InternVL3-8B': (-80, -10, 'right'),
    'InternVL-2.5-78B': (-80, 0, 'right'),
    'InternVL-2.5-8B': (-8, -20, 'right'),
    # Llama models
    'Llama-3.2-90B-Vision-Instruct': (45, 0, 'left'),
    'Llama-3.2-11B-Vision-Instruct': (25, 0, 'left'),
    # Gemma models
    'gemma-3-27b-it': (-80, 10, 'right'),
    'gemma-3-12b-it': (-60, 30, 'right')
}

# Add model identifier text with custom positioning and family colors - flipped axes
for model_name, (accuracy, agreement, family) in models_data.items():
    identifier = get_model_identifier(model_name)
    if identifier and model_name in label_offsets:
        offset_x, offset_y, ha = label_offsets[model_name]
        color = family_colors[family]
        
        # Add leader lines for models with far positioning
        if model_name in ['GPT-5', 'GPT-5-mini', 'GPT-5-nano', 'Claude-4.1-Opus', 'Claude-4-Sonnet', 'InternVL-2.5-78B', 'InternVL-2.5-8B', 'InternVL3-78B', 'InternVL3-8B', 'Llama-3.2-90B-Vision-Instruct', 'Llama-3.2-11B-Vision-Instruct', 'Qwen2.5-VL-72B-Instruct', 'Qwen2.5-VL-7B-Instruct', 'Qwen2.5-Omni-7B', 'gemma-3-27b-it', 'gemma-3-12b-it']:
            ax.annotate(identifier, (accuracy, agreement), xytext=(offset_x, offset_y), 
                       textcoords='offset points', fontsize=25, fontweight='bold',
                       color=color, alpha=0.9, ha=ha, va='center',
                       arrowprops=dict(arrowstyle='-', color='grey', alpha=0.3, linewidth=0.8))
        else:
            ax.annotate(identifier, (accuracy, agreement), xytext=(offset_x, offset_y), 
                       textcoords='offset points', fontsize=25, fontweight='bold',
                       color=color, alpha=0.9, ha=ha, va='center')

# No direct labels on plot - models will be identified through legend connections

# Customize plot - flipped axis labels
ax.set_xlabel('Average Accuracy', fontsize=28, labelpad=20)
ax.set_ylabel('Language-Vision Agreement', fontsize=28, labelpad=20)

# Set axis limits with better padding and more vertical space to show improvement potential
ax.set_xlim(0.23, 0.84)
ax.set_ylim(0.24, 0.80)

# Add random baseline text annotation
ax.text(0.7, 0.42, 'Random Baseline', fontsize=26, color='#666666', 
        ha='center', va='bottom', weight='bold')

# Model names removed as requested - to be added manually later

# Increase tick label font sizes and format axes
ax.tick_params(axis='both', which='major', labelsize=22)

# Format axis to show percentages properly
from matplotlib.ticker import FuncFormatter

def to_percent(y, position):
    return f'{int(y*100)}%'

ax.xaxis.set_major_formatter(FuncFormatter(to_percent))
ax.yaxis.set_major_formatter(FuncFormatter(to_percent))

# No abbreviation key needed since legend shows full model names

# No title/caption as requested

plt.tight_layout()
plt.savefig('/datadrive/josephtang/SEAM/code/plots/acc_vs_agr.png', dpi=300, bbox_inches='tight')
plt.show()