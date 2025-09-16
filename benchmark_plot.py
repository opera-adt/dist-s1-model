import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# --- 1. Load the Data ---
try:
    df = pd.read_csv('benchmark_raw_results.csv')
except FileNotFoundError:
    print("Error: 'data.csv' not found. Please save your data in a file with that name.")
    exit()

# --- 2. Create Combined Metric Columns ---
df['nll'] = df['test_nll'] + df['train_nll']
df['mse'] = df['test_mse'] + df['train_mse']
print("Created combined 'nll' and 'mse' columns.")

# --- 3. Average Metrics Across Datasets ---
metrics_to_average = ['nll', 'mse', 'run_time_sec', 'peak_gpu_memory_mb']
# Group by model and sequence length, then calculate the mean for all numeric columns.
averaged_df = df.groupby(['model', 'target_seq_len'])[metrics_to_average].mean().reset_index()
print("Averaged metrics across all datasets.")


# --- 4. Identify Unique Models and Define Metrics to Plot ---
unique_models = averaged_df['model'].unique()
metrics_to_plot = metrics_to_average


# --- 5. Generate a Single, Averaged Plot Figure ---
colors = cm.get_cmap('tab10', len(unique_models)) 

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

fig.suptitle('Overall Model Comparison (Averaged Across All Datasets)', fontsize=20, y=0.97)

# This is where we'll apply the specific logic for the memory plot
for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]
    
    # Counter for applying a unique offset to each constant horizontal line
    offset_counter = 0
    # The amount of memory (in MB) to shift each overlapping line
    offset_step = 50 

    for model_idx, (model_name, group) in enumerate(averaged_df.groupby('model')):
        group = group.sort_values('target_seq_len')
        
        # Get the y-values to plot
        y_values = group[metric]
        
        # --- NEW LOGIC: Apply offset only to the memory plot for constant lines ---
        # We check if the metric is 'peak_gpu_memory_mb' and if the line is flat
        # (i.e., it only has 1 unique value).
        if metric == 'peak_gpu_memory_mb' and group[metric].nunique() == 1:
            # Apply the vertical offset
            y_values = y_values + (offset_counter * offset_step)
            offset_counter += 1
        # --- End of new logic ---

        ax.plot(
            group['target_seq_len'], 
            y_values, # Plot the original or offset y_values
            marker='o', 
            linestyle='-', 
            label=model_name,
            color=colors(model_idx)
        )
        
    # Set titles and labels
    title = metric.replace('_', ' ').title()
    if metric in ['nll', 'mse']:
         title += ' (Test + Train)'
    
    # Add a note to the memory plot title for clarity
    if metric == 'peak_gpu_memory_mb':
        title += ' (Offsets Added for Clarity)'
            
    ax.set_title(title)
    ax.set_xlabel('Target Sequence Length')
    ax.set_ylabel(f'Average {title.split(" (")[0]}') # Use a cleaner Y-label
    ax.grid(True, linestyle='--', alpha=0.6)

# Create a single legend for the entire figure
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.92))

plt.tight_layout(rect=[0, 0, 1, 0.94])

# Save and show the final plot
output_filename = 'final_averaged_model_comparison_with_offset.png'
plt.savefig(output_filename)
print(f"Saved final averaged plot to {output_filename}")
plt.show()

plt.close(fig)

print("\nAnalysis complete.")