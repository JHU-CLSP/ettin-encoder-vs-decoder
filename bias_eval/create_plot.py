import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle

# Set the style for scientific publication
plt.style.use('seaborn-v0_8-white')
sns.set_palette("husl")

def parse_model_data(json_file_path):
    """Parse the JSON data and extract model information"""
    
    with open(json_file_path, 'r') as f:
        models = json.load(f)
    
    model_info = []
    
    for key, model_data in models.items():
        model_name = model_data['model_name']
        pred_dist = model_data['prediction_distribution']
        
        # Extract model type and size
        if 'ettin-encoder' in model_name and 'ettin-enc-from-dec' not in model_name:
            model_type = 'Ettin-Encoder'
            size_match = model_name.split('ettin-encoder-')[1]
            model_size = size_match
        elif 'ettin-decoder' in model_name:
            model_type = 'Ettin-Decoder'
            size_match = model_name.split('ettin-decoder-')[1]
            model_size = size_match
        elif 'encoder_from_decoder' in model_name or 'ettin-enc-from-dec' in model_name:
            model_type = 'Encoder-from-Decoder'
            if 'encoder_from_decoder' in model_name:
                size_match = model_name.split('encoder_from_decoder_')[1].split('-cross-train')[0]
            else:  # ettin-enc-from-dec
                size_match = model_name.split('ettin-enc-from-dec-')[1]
            model_size = size_match
        elif 'ettin-dec-from-enc' in model_name:
            model_type = 'Decoder-from-Encoder'
            size_match = model_name.split('ettin-dec-from-enc-')[1]
            model_size = size_match
        else:
            continue
        
        model_info.append({
            'type': model_type,
            'size': model_size,
            'male': pred_dist.get('male', 0),
            'female': pred_dist.get('female', 0),
            'neutral': pred_dist.get('neutral', 0)
        })
    
    return model_info

def create_visualization(model_info, output_file='model_prediction_distribution.png'):
    """Create a stacked bar plot showing prediction distributions"""
    
    # Define size ordering for proper visualization
    ettin_sizes = ['17m', '32m', '66m', '150m', '400m', '1b']
    enc_from_dec_sizes = ['very_tiny', 'tiny', 'mini', 'base', 'large', 'huge']
    
    # Create a consistent size mapping for x-axis labels
    consistent_labels = ['17M', '32M', '66M', '150M', '400M', '1B']
    enc_labels = ['Very Tiny', 'Tiny', 'Mini', 'Base', 'Large', 'Huge']
    
    # Separate data by model type
    ettin_data = [m for m in model_info if m['type'] == 'Ettin-Encoder']
    ettin_dec_data = [m for m in model_info if m['type'] == 'Ettin-Decoder']
    enc_from_dec_data = [m for m in model_info if m['type'] == 'Encoder-from-Decoder']
    dec_from_enc_data = [m for m in model_info if m['type'] == 'Decoder-from-Encoder']
    
    # Split encoder-from-decoder by size format
    enc_from_dec_ettin = [m for m in enc_from_dec_data if m['size'] in ettin_sizes]
    
    def pad_data_to_sizes(data, expected_sizes):
        """Pad data to match expected sizes, filling missing with zeros"""
        padded_data = []
        existing_sizes = {d['size']: d for d in data}
        
        for size in expected_sizes:
            if size in existing_sizes:
                padded_data.append(existing_sizes[size])
            else:
                # Add missing size with zero values
                padded_data.append({
                    'type': data[0]['type'] if data else 'Unknown',
                    'size': size,
                    'male': 0,
                    'female': 0,
                    'neutral': 0
                })
        return padded_data
    
    # Sort by size and pad missing data
    ettin_data.sort(key=lambda x: ettin_sizes.index(x['size']))
    ettin_data = pad_data_to_sizes(ettin_data, ettin_sizes)
    
    ettin_dec_data.sort(key=lambda x: ettin_sizes.index(x['size']))
    ettin_dec_data = pad_data_to_sizes(ettin_dec_data, ettin_sizes)
    
    enc_from_dec_ettin.sort(key=lambda x: ettin_sizes.index(x['size']))
    enc_from_dec_ettin = pad_data_to_sizes(enc_from_dec_ettin, ettin_sizes)
    
    dec_from_enc_data.sort(key=lambda x: ettin_sizes.index(x['size']) if x['size'] in ettin_sizes else 0)
    dec_from_enc_data = pad_data_to_sizes(dec_from_enc_data, ettin_sizes)
    
    # Create the figure with subplots (4 plots: ettin-encoder, ettin-decoder, enc-from-dec original, enc-from-dec ettin)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
    
    # Define colors for the categories
    colors = {'male': '#2E86AB', 'female': '#A23B72', 'neutral': '#F18F01'}
    
    def plot_stacked_bars(ax, data, model_type, x_labels=None):
        """Helper function to create stacked bars for each model type"""
        
        if x_labels is None:
            x_labels = consistent_labels
        
        # Prepare data for plotting
        male_vals = [d['male'] for d in data]
        female_vals = [d['female'] for d in data]
        neutral_vals = [d['neutral'] for d in data]
        
        # Create stacked bars
        x = np.arange(len(x_labels))
        width = 0.6
        
        p1 = ax.bar(x, male_vals, width, label='Male', color=colors['male'], alpha=0.8)
        p2 = ax.bar(x, female_vals, width, bottom=male_vals, label='Female', color=colors['female'], alpha=0.8)
        p3 = ax.bar(x, neutral_vals, width, bottom=np.array(male_vals) + np.array(female_vals), 
                   label='Neutral', color=colors['neutral'], alpha=0.8)
        
        # Customize the subplot
        ax.set_title(f'{model_type}', fontsize=14, pad=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=14)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.tick_params(axis='y', labelsize=12)
        ax.set_ylim(0, 1.05)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        # Add percentage labels on bars (only for non-zero values)
        for i, (male, female, neutral) in enumerate(zip(male_vals, female_vals, neutral_vals)):
            # Only show labels for segments > 5% and non-zero
            if male > 0.07:
                ax.text(i, male/2, f'{male:.0%}', ha='center', va='center', 
                       fontweight='bold', fontsize=9, color='white')
            if female > 0.07:
                ax.text(i, male + female/2, f'{female:.0%}', ha='center', va='center', 
                       fontweight='bold', fontsize=9, color='white')
            if neutral > 0.07:
                ax.text(i, male + female + neutral/2, f'{neutral:.0%}', ha='center', va='center', 
                       fontweight='bold', fontsize=9, color='white')
        
        return ax
    
    # Plot all model types
    plot_stacked_bars(ax1, ettin_data, 'Ettin-Encoder')
    plot_stacked_bars(ax2, ettin_dec_data, 'Ettin-Decoder')
    plot_stacked_bars(ax3, enc_from_dec_data, 'Encoder-from-Decoder')
    plot_stacked_bars(ax4, dec_from_enc_data, 'Decoder-from-Encoder')
    
    # Add shared y-axis label for the entire figure
    fig.supylabel('Prediction Probability', fontsize=20, x=0.02)
    
    # Add a single legend for the entire figure - horizontal at the top
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.54, 1.04), 
               fontsize=14, frameon=True, fancybox=True, shadow=True, ncol=3)
    
    # Add model size progression annotation
    ax4.set_xlabel('Model Size →', fontsize=18)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.08, hspace=0.4)
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def print_summary_statistics(model_info):
    """Print summary statistics for the analysis"""
    
    print("=== Model Prediction Distribution Analysis ===\n")
    
    for model_type in ['Ettin-Encoder', 'Ettin-Decoder', 'Encoder-from-Decoder', 'Decoder-from-Encoder']:
        data = [m for m in model_info if m['type'] == model_type]
        
        if not data:
            continue
            
        print(f"{model_type} Models:")
        print(f"  Number of models: {len(data)}")
        
        # Calculate averages
        avg_male = np.mean([d['male'] for d in data])
        avg_female = np.mean([d['female'] for d in data])
        avg_neutral = np.mean([d['neutral'] for d in data])
        
        print(f"  Average Male prediction: {avg_male:.3f} ({avg_male:.1%})")
        print(f"  Average Female prediction: {avg_female:.3f} ({avg_female:.1%})")
        print(f"  Average Neutral prediction: {avg_neutral:.3f} ({avg_neutral:.1%})")
        
        # Find trends
        male_vals = [d['male'] for d in data]
        if len(male_vals) > 1:
            trend = "increasing" if male_vals[-1] > male_vals[0] else "decreasing"
            print(f"  Male prediction trend with size: {trend}")
        print()

# Main execution
if __name__ == "__main__":
    # Parse the data
    model_info = parse_model_data("bias_evaluation_results.json")
    
    # Print summary statistics
    print_summary_statistics(model_info)
    
    # Create the visualization
    create_visualization(model_info)
    
    print("Visualization saved as 'model_prediction_distribution.png'")
    print("\nNote: The plot shows the distribution of model predictions across three categories:")
    print("- Male (blue): Predictions classified as male")
    print("- Female (pink): Predictions classified as female") 
    print("- Neutral (orange): Predictions classified as neutral")
    print("Each bar represents a model size, arranged from smallest to largest.")