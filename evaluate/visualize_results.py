import os
import ast
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import argparse
import json

# 9个主题文件夹的路径
ALL_THEMES = ['Career', 'Education', 'Emotion&Stress', 'Family Relationship', 'Love&Marriage', 
              'Mental Disease', 'Self-growth', 'Sex', 'Social Relationship']

def parse_score_text(score_text: str) -> Optional[List[int]]:
    """
    Parse the score text returned by the evaluation model and extract a valid score list.
    Returns None if no valid score could be extracted.
    
    Args:
        score_text: The text response from the model containing the scores
        
    Returns:
        A list of 4 integers representing the scores, or None if parsing failed
    """
    # Remove any leading/trailing whitespace
    score_text = score_text.strip()
    
    # Method 1: Try direct parsing with ast.literal_eval
    try:
        parsed = ast.literal_eval(score_text)
        if isinstance(parsed, list) and len(parsed) == 4 and all(isinstance(x, int) for x in parsed):
            return parsed
    except (SyntaxError, ValueError):
        pass  # Continue to next method if this fails
    
    # Method 2: Use regex to find a list pattern [n, n, n, n] anywhere in the text
    pattern = r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]'
    match = re.search(pattern, score_text)
    if match:
        return [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
    
    # Method 3: Try to find four consecutive numbers that might represent scores
    number_pattern = r'(\d+)'
    matches = re.findall(number_pattern, score_text)
    if len(matches) >= 4:
        # Use the first four numbers found
        potential_scores = [int(matches[i]) for i in range(4)]
        # Validate the scores are within expected ranges
        if (0 <= potential_scores[0] <= 2 and  # Comprehensiveness: 0-2
            0 <= potential_scores[1] <= 3 and  # Professionalism: 0-3
            0 <= potential_scores[2] <= 3 and  # Authenticity: 0-3
            0 <= potential_scores[3] <= 1):    # Safety: 0-1
            return potential_scores
    
    # If all methods fail, return None
    print(f"Warning: Could not parse score: {score_text}")
    return None

def get_available_model_dirs() -> List[str]:
    """
    Get all available model directories in the results folder.
    
    Returns:
        A list of model directory names
    """
    base_dir = os.path.abspath(os.path.join('..'))
    results_base_dir = os.path.join(base_dir, "Results_Turn_Based_Dialogue_Evaluation")
    
    if not os.path.exists(results_base_dir):
        print(f"No results directory found at: {results_base_dir}")
        return []
    
    # Get all directories from results folder
    all_dirs = [d for d in os.listdir(results_base_dir) 
               if os.path.isdir(os.path.join(results_base_dir, d))]
    
    # Extract unique reply model names from the combined directory names
    reply_models = set()
    for d in all_dirs:
        if "_evaluated_by_" in d:
            reply_model = d.split("_evaluated_by_")[0]
            reply_models.add(reply_model)
    
    return sorted(list(reply_models))

def get_evaluation_models(reply_model: str) -> List[str]:
    """
    Get all available evaluation models for a given reply model.
    
    Args:
        reply_model: The reply model directory name
    
    Returns:
        A list of evaluation model directory names
    """
    base_dir = os.path.abspath(os.path.join('..'))
    results_base_dir = os.path.join(base_dir, "Results_Turn_Based_Dialogue_Evaluation")
    
    if not os.path.exists(results_base_dir):
        print(f"No directory found for reply model: {reply_model}")
        return []
    
    # Find all directories that match the pattern "{reply_model}_evaluated_by_{eval_model}"
    all_dirs = [d for d in os.listdir(results_base_dir) 
               if os.path.isdir(os.path.join(results_base_dir, d))]
    
    eval_models = []
    for d in all_dirs:
        if d.startswith(reply_model + "_evaluated_by_"):
            eval_model = d.split("_evaluated_by_")[1]
            eval_models.append(eval_model)
    
    return eval_models

def calculate_theme_averages(theme_folder: str, reply_model: str, eval_model: str = "default") -> List[float]:
    """
    Calculate average scores for each metric across all evaluation files for a theme.
    
    Args:
        theme_folder: The name of the theme folder
        reply_model: The reply model directory name
        eval_model: The evaluation model directory name (default: "default")
        
    Returns:
        A list of 4 floats representing the average scores for each metric
    """
    base_dir = os.path.abspath(os.path.join('..'))
    results_base_dir = os.path.join(base_dir, "Results_Turn_Based_Dialogue_Evaluation")
    
    # Create the full directory name using the pattern
    full_model_dir = f"{reply_model}_evaluated_by_{eval_model}"
    results_dir = os.path.join(results_base_dir, full_model_dir, theme_folder)
    
    if not os.path.exists(results_dir):
        print(f"No results directory found for theme: {theme_folder}, reply model: {reply_model}, eval model: {eval_model}")
        return [0, 0, 0, 0]  # Return zeros if no data available
    
    # Get all evaluation result files
    result_files = [f for f in os.listdir(results_dir) if f.startswith("evaluation_results_") and f.endswith(".txt")]
    
    if not result_files:
        print(f"No evaluation files found for theme: {theme_folder} in reply model: {reply_model}, eval model: {eval_model}")
        return [0, 0, 0, 0]  # Return zeros if no files found
    
    all_scores = []
    
    # Read and process each file
    for file in result_files:
        file_path = os.path.join(results_dir, file)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Round") and "Score:" in line:
                    # Extract score list from the line
                    score_part = line.split("Score:")[1].strip()
                    try:
                        score = ast.literal_eval(score_part)
                        if score is not None and len(score) == 4:
                            all_scores.append(score)
                    except (SyntaxError, ValueError):
                        continue
    
    if not all_scores:
        print(f"No valid scores found in files for theme: {theme_folder} in reply model: {reply_model}, eval model: {eval_model}")
        return [0, 0, 0, 0]
    
    # Calculate average for each metric
    avg_scores = [round(sum(col) / len(col), 2) for col in zip(*all_scores)]
    return avg_scores

def plot_radar_charts(theme_averages: Dict[str, List[float]], reply_model: str, eval_model: str = "default", 
                     selected_metrics: List[int] = None, output_dir: str = None):
    """
    Create separate radar charts for each evaluation metric across all themes.
    Each metric gets its own figure file.
    
    Args:
        theme_averages: Dictionary mapping theme names to their average scores
        reply_model: Name of the reply model
        eval_model: Name of the evaluation model
        selected_metrics: List of indices for the metrics to include (None for all)
        output_dir: Directory to save the output file
    """
    # Setup radar chart
    themes = list(theme_averages.keys())
    metric_names = ['Comprehensiveness', 'Professionalism', 'Authenticity', 'Safety']
    
    # Use selected metrics or all metrics
    if selected_metrics is None:
        selected_metrics = list(range(len(metric_names)))
    
    # Determine number of themes (vertices)
    N = len(themes)
    
    if N == 0:
        print("No themes to plot.")
        return
    
    # Create angle for each theme
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Create model-specific output directory
    model_output_dir = output_dir
    if output_dir is None:
        model_output_dir = f"{reply_model}_{eval_model}_charts"
    else:
        model_output_dir = os.path.join(output_dir, f"{reply_model}_{eval_model}_charts")
    
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    
    # Color map for lines - different color for each metric
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    
    # For each metric, create a separate radar chart in its own figure
    for i in range(4):
        # Skip metrics not in selected_metrics
        if i not in selected_metrics:
            continue
            
        # Create a new figure for this metric
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        metric_name = metric_names[i]
        ax.set_title(f"{metric_name} - {reply_model}\nEvaluated by {eval_model}", size=14, y=1.05)
        
        # Set radar chart angles and labels
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Set theme labels as tick labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(themes, fontsize=8)
        
        # Define y-axis limits based on metric type
        if metric_name == 'Comprehensiveness':
            ax.set_ylim(0, 2)
            ax.set_yticks(np.arange(0, 2.1, 0.5))
        elif metric_name in ['Professionalism', 'Authenticity']:
            ax.set_ylim(0, 3)
            ax.set_yticks(np.arange(0, 3.1, 0.5))
        else:  # Safety
            ax.set_ylim(0, 1)
            ax.set_yticks(np.arange(0, 1.1, 0.2))
        
        # Get the scores for this metric across all themes
        values = [theme_averages[theme][i] for theme in themes]
        values += values[:1]  # Close the loop
        
        # Plot the values - use the metric's specific color
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=metric_name)
        ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Add a legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        
        # Save this metric's figure
        output_file = os.path.join(model_output_dir, f"{metric_name.lower()}_radar_chart.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved {metric_name} chart to '{output_file}'")
        plt.close(fig)
    
    print(f"All charts saved to directory: {model_output_dir}")

def plot_metric_comparison(model_dirs: List[str], themes: List[str], metric_idx: int, metric_name: str, output_dir: str = None):
    """
    Plot a comparison of a specific metric across different models for all themes.
    
    Args:
        model_dirs: List of model directories to compare
        themes: List of themes to include
        metric_idx: Index of the metric to compare (0-3)
        metric_name: Name of the metric (for title and legend)
        output_dir: Directory to save the output file
    """
    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define bar width and positions
    n_models = len(model_dirs)
    n_themes = len(themes)
    bar_width = 0.8 / n_models
    
    # Define colors for models
    colors = plt.cm.viridis(np.linspace(0, 1, n_models))
    
    # For each model, plot bars for the selected metric across all themes
    for i, model_dir in enumerate(model_dirs):
        # Get metric values for each theme
        values = []
        for theme in themes:
            avg_scores = calculate_theme_averages(theme, model_dir)
            values.append(avg_scores[metric_idx])
        
        # Calculate bar positions
        # positions = np.arange(n_themes) + (i - n_models/2 + 0.5) * bar_width
        
        # Plot bars
        # ax.bar(positions, values, bar_width, label=model_dir, color=colors[i], alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Themes')
    if metric_name == 'Comprehensiveness':
        ax.set_ylabel('Score (0-2)')
        ax.set_ylim(0, 2)
    elif metric_name in ['Professionalism', 'Authenticity']:
        ax.set_ylabel('Score (0-3)')
        ax.set_ylim(0, 3)
    else:  # Safety
        ax.set_ylabel('Score (0-1)')
        ax.set_ylim(0, 1)
    
    ax.set_title(f'{metric_name} Comparison Across Models')
    ax.set_xticks(np.arange(n_themes))
    ax.set_xticklabels(themes, rotation=45, ha='right')
    ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save the figure
    if output_dir is None:
        output_file = f"comparison_{metric_name}.png"
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f"comparison_{metric_name}.png")
        
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {metric_name} comparison to '{output_file}'")
    plt.close()

def interactive_selection(options: List[str], prompt_message: str, allow_multiple: bool = False) -> List[int]:
    """
    Interactively prompt the user to select options from a list.
    
    Args:
        options: List of options to select from
        prompt_message: Message to display to the user
        allow_multiple: Whether to allow selecting multiple options
        
    Returns:
        List of selected indices
    """
    if not options:
        print("No options available.")
        return []
    
    # Display options
    print(prompt_message)
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    
    if allow_multiple:
        print("\nEnter comma-separated numbers, or 'all' for all options:")
    else:
        print("\nEnter a number:")
    
    user_input = input("> ").strip()
    
    if allow_multiple and user_input.lower() == 'all':
        return list(range(len(options)))
    
    try:
        if allow_multiple:
            indices = [int(idx.strip()) - 1 for idx in user_input.split(',')]
            selected = [idx for idx in indices if 0 <= idx < len(options)]
            if not selected:
                print("No valid options selected. Using default.")
                return []
            return selected
        else:
            idx = int(user_input) - 1
            if 0 <= idx < len(options):
                return [idx]
            else:
                print("Invalid selection. Using default.")
                return []
    except ValueError:
        print("Invalid input. Using default.")
        return []

def compare_models_radar_charts(models_data: Dict[str, Dict[str, List[float]]], eval_model: str,
                               selected_metrics: List[int] = None, output_dir: str = None):
    """
    Create radar charts comparing multiple reply models for each evaluation metric.
    
    Args:
        models_data: Dictionary mapping reply model names to their theme averages dictionaries
        eval_model: Name of the evaluation model used
        selected_metrics: List of indices for the metrics to include (None for all)
        output_dir: Directory to save the output file
    """
    if not models_data:
        print("No data to plot.")
        return
    
    # Setup metrics and themes (assuming all models have the same themes)
    metric_names = ['Comprehensiveness', 'Professionalism', 'Authenticity', 'Safety']
    first_model = next(iter(models_data.values()))
    themes = list(first_model.keys())
    
    # Use selected metrics or all metrics
    if selected_metrics is None:
        selected_metrics = list(range(len(metric_names)))
    
    # Determine number of themes (vertices)
    N = len(themes)
    
    if N == 0:
        print("No themes to plot.")
        return
    
    # Create angle for each theme
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Create evaluation model-specific output directory
    eval_output_dir = output_dir
    if output_dir is None:
        eval_output_dir = f"{eval_model}_charts"
    else:
        eval_output_dir = os.path.join(output_dir, f"{eval_model}_charts")
    
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    
    # Use a color map for different reply models
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))
    
    # For each metric, create a separate radar chart
    for i in range(4):
        # Skip metrics not in selected_metrics
        if i not in selected_metrics:
            continue
            
        # Create a new figure for this metric
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        metric_name = metric_names[i]
        
        # Set radar chart angles and labels
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Set theme labels as tick labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(themes, fontsize=8)
        
        # Define y-axis limits based on metric type
        if metric_name == 'Comprehensiveness':
            ax.set_ylim(0, 2)
            ax.set_yticks(np.arange(0, 2.1, 0.5))
        elif metric_name in ['Professionalism', 'Authenticity']:
            ax.set_ylim(0, 3)
            ax.set_yticks(np.arange(0, 3.1, 0.5))
        else:  # Safety
            ax.set_ylim(0, 1)
            ax.set_yticks(np.arange(0, 1.1, 0.2))
        
        # Plot each model's data
        for j, (model_name, theme_data) in enumerate(models_data.items()):
            # Get the scores for this metric across all themes
            values = [theme_data[theme][i] for theme in themes]
            values += values[:1]  # Close the loop
            
            # Plot the values with a different color for each model
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[j], label=model_name)
            ax.fill(angles, values, alpha=0.1, color=colors[j])
        
        # Add a legend for the models
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        
        plt.tight_layout()
        
        # Save this metric's figure
        output_file = os.path.join(eval_output_dir, f"{metric_name.lower()}_radar_chart.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved {metric_name} comparison chart to '{output_file}'")
        plt.close(fig)
    
    print(f"All comparison charts saved to directory: {eval_output_dir}")

def main():
    """
    Main function to handle command-line arguments and generate visualizations.
    """
    parser = argparse.ArgumentParser(description='Generate radar charts from evaluation results')
    parser.add_argument('--model-dir', type=str, help='Specific model directory to visualize')
    parser.add_argument('--compare', action='store_true', help='Compare multiple models')
    parser.add_argument('--output-dir', type=str, default='visualizations', help='Directory to save output files')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode with more selection options')
    
    args = parser.parse_args()
    
    # Get available model directories
    reply_models = get_available_model_dirs()
    
    if not reply_models:
        print("No evaluation results found. Please run the evaluation script first.")
        return
    
    # Create output directory if it doesn't exist
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Interactive mode with enhanced selection capabilities
    if args.interactive or (not args.model_dir and not args.compare):
        # 1. Get all evaluation models used across reply models
        all_eval_models = set()
        for reply_model in reply_models:
            eval_models = get_evaluation_models(reply_model)
            all_eval_models.update(eval_models)
        
        eval_model_list = sorted(list(all_eval_models))
        
        print("=== Select Evaluation Model ===")
        selected_eval_indices = interactive_selection(eval_model_list, "Available evaluation models:", allow_multiple=False)
        
        if not selected_eval_indices:
            print("No valid evaluation model selected. Exiting.")
            return
        
        selected_eval_model = eval_model_list[selected_eval_indices[0]]
        
        # 2. Find reply models that were evaluated by this evaluation model
        compatible_reply_models = [model for model in reply_models 
                                 if selected_eval_model in get_evaluation_models(model)]
        
        print("\n=== Select Reply Models to Compare ===")
        selected_reply_indices = interactive_selection(compatible_reply_models, "Available reply models:", allow_multiple=True)
        
        if not selected_reply_indices:
            print("No valid reply models selected. Exiting.")
            return
        
        selected_reply_models = [compatible_reply_models[idx] for idx in selected_reply_indices]
        
        # 3. Select metrics to visualize
        metric_names = ['Comprehensiveness', 'Professionalism', 'Authenticity', 'Safety']
        print("\n=== Select Metrics to Visualize ===")
        selected_metric_indices = interactive_selection(metric_names, "Available metrics:", allow_multiple=True)
        
        if not selected_metric_indices:
            print("No valid metrics selected. Using all.")
            selected_metric_indices = list(range(len(metric_names)))
        
        # 4. Allow user to choose between text files or JSON data
        print("\n=== Select Data Source ===")
        data_sources = ['Text Files', 'JSON Data']
        data_source_idx = interactive_selection(data_sources, "Select data source:", allow_multiple=False)
        
        # Load data for all selected reply models
        models_data = {}
        for reply_model in selected_reply_models:
            if data_source_idx and data_source_idx[0] == 1:  # JSON Data selected
                theme_averages = load_json_theme_averages(reply_model, selected_eval_model)
                if theme_averages:
                    models_data[reply_model] = theme_averages
                else:
                    print(f"No JSON data found for {reply_model} with evaluation model {selected_eval_model}")
            else:  # Default to Text Files
                theme_averages = {}
                for theme in ALL_THEMES:
                    avg_scores = calculate_theme_averages(theme, reply_model, selected_eval_model)
                    theme_averages[theme] = avg_scores
                models_data[reply_model] = theme_averages
        
        # Generate comparison radar charts
        if models_data:
            compare_models_radar_charts(models_data, selected_eval_model, selected_metric_indices, args.output_dir)
        else:
            print("No data available for the selected models and evaluation model.")
    
    # Original functionality for --model-dir and --compare options
    elif args.model_dir:
        # Visualize a specific model directory
        if args.model_dir not in reply_models:
            print(f"Model directory '{args.model_dir}' not found. Available directories:")
            for model_dir in reply_models:
                print(f"- {model_dir}")
            return
        
        # Calculate averages for each theme
        theme_averages = {}
        for theme in ALL_THEMES:
            avg_scores = calculate_theme_averages(theme, args.model_dir)
            theme_averages[theme] = avg_scores
        
        # Plot radar charts
        plot_radar_charts(theme_averages, args.model_dir, output_dir=args.output_dir)
    
    elif args.compare:
        # Compare multiple models
        print("Available model directories:")
        for i, model_dir in enumerate(reply_models, 1):
            print(f"{i}. {model_dir}")
        
        print("\nPlease select models to compare (comma-separated numbers, or 'all' for all models):")
        user_input = input("> ").strip()
        
        if user_input.lower() == 'all':
            selected_models = reply_models
        else:
            try:
                selected_indices = [int(idx.strip()) - 1 for idx in user_input.split(',')]
                selected_models = [reply_models[idx] for idx in selected_indices if 0 <= idx < len(reply_models)]
                if not selected_models:
                    print("No valid models selected. Exiting.")
                    return
            except ValueError:
                print("Invalid input. Please enter comma-separated numbers or 'all'. Exiting.")
                return
        
        # Find common evaluation models for the selected reply models
        common_eval_models = set()
        for model in selected_models:
            eval_models = get_evaluation_models(model)
            if not common_eval_models:
                common_eval_models = set(eval_models)
            else:
                common_eval_models &= set(eval_models)
        
        if not common_eval_models:
            print("No common evaluation models found for the selected reply models.")
            return
        
        print("\nPlease select an evaluation model to use for comparison:")
        for i, eval_model in enumerate(sorted(common_eval_models), 1):
            print(f"{i}. {eval_model}")
        
        eval_model_input = input("> ").strip()
        try:
            eval_model_idx = int(eval_model_input) - 1
            sorted_eval_models = sorted(common_eval_models)
            if 0 <= eval_model_idx < len(sorted_eval_models):
                selected_eval_model = sorted_eval_models[eval_model_idx]
            else:
                print("Invalid selection. Using the first evaluation model.")
                selected_eval_model = sorted_eval_models[0]
        except ValueError:
            print("Invalid input. Using the first evaluation model.")
            selected_eval_model = sorted(common_eval_models)[0]
        
        # Load data for all selected models with the chosen evaluation model
        models_data = {}
        for model in selected_models:
            theme_averages = {}
            for theme in ALL_THEMES:
                avg_scores = calculate_theme_averages(theme, model, selected_eval_model)
                theme_averages[theme] = avg_scores
            models_data[model] = theme_averages
        
        # Generate comparison radar charts
        compare_models_radar_charts(models_data, selected_eval_model, output_dir=args.output_dir)

def load_json_theme_averages(reply_model: str, eval_model: str) -> Dict[str, List[float]]:
    """
    Load theme averages from the JSON data generated by the evaluation.
    
    Args:
        reply_model: The name of the reply model
        eval_model: The name of the evaluation model
        
    Returns:
        A dictionary mapping theme names to their average scores
    """
    base_dir = os.path.abspath(os.path.join('..'))
    json_path = os.path.join(base_dir, "site", "assets", "dataset_averages.json")
    
    if not os.path.exists(json_path):
        print(f"JSON data file not found at: {json_path}")
        return {}
    
    try:
        with open(json_path, 'r') as f:
            dataset_averages = json.load(f)
        
        # Filter results for the specific models
        theme_averages = {}
        for entry in dataset_averages:
            if entry["reply_model"] == reply_model.replace('_', '/') and entry["eval_model"] == eval_model.replace('_', '/'):
                theme = entry["theme"]
                # Convert the metrics dictionary to a list in the expected order
                metrics_list = [
                    entry["metrics"]["comprehensiveness"],
                    entry["metrics"]["professionalism"],
                    entry["metrics"]["authenticity"],
                    entry["metrics"]["safety"]
                ]
                theme_averages[theme] = metrics_list
        
        return theme_averages
    
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"Error loading JSON data: {e}")
        return {}

def plot_json_comparison(model_dirs: List[str], themes: List[str], metric_name: str, output_dir: str = None):
    """
    Plot a comparison of a specific metric across different models using JSON data.
    
    Args:
        model_dirs: List of model directories to compare
        themes: List of themes to include
        metric_name: Name of the metric (for title and legend)
        output_dir: Directory to save the output file
    """
    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get the metric index
    metric_names = ['Comprehensiveness', 'Professionalism', 'Authenticity', 'Safety']
    metric_keys = ['comprehensiveness', 'professionalism', 'authenticity', 'safety']
    try:
        metric_idx = metric_names.index(metric_name)
        metric_key = metric_keys[metric_idx]
    except ValueError:
        print(f"Unknown metric: {metric_name}")
        return
    
    # Load JSON data
    base_dir = os.path.abspath(os.path.join('..'))
    json_path = os.path.join(base_dir, "site", "assets", "dataset_averages.json")
    
    if not os.path.exists(json_path):
        print(f"JSON data file not found at: {json_path}")
        return
    
    try:
        with open(json_path, 'r') as f:
            dataset_averages = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading JSON data: {e}")
        return
    
    # Define bar width and positions
    n_models = len(model_dirs)
    n_themes = len(themes)
    bar_width = 0.8 / n_models
    
    # Define colors for models
    colors = plt.cm.viridis(np.linspace(0, 1, n_models))
    
    # For each model, plot bars for the selected metric across all themes
    for i, model_dir in enumerate(model_dirs):
        # Get metric values for each theme from JSON data
        values = []
        for theme in themes:
            # Find matching entry in dataset_averages
            matching = [entry for entry in dataset_averages 
                        if entry["reply_model"] == model_dir and entry["theme"] == theme]
            
            if matching:
                # Use the first matching entry (should be only one)
                value = matching[0]["metrics"][metric_key]
                values.append(value)
            else:
                values.append(0)  # Use zero for missing data
        
        # Calculate bar positions
        # positions = np.arange(n_themes) + (i - n_models/2 + 0.5) * bar_width
        
        # Plot bars
        # ax.bar(positions, values, bar_width, label=model_dir, color=colors[i], alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Themes')
    if metric_name == 'Comprehensiveness':
        ax.set_ylabel('Score (0-2)')
        ax.set_ylim(0, 2)
    elif metric_name in ['Professionalism', 'Authenticity']:
        ax.set_ylabel('Score (0-3)')
        ax.set_ylim(0, 3)
    else:  # Safety
        ax.set_ylabel('Score (0-1)')
        ax.set_ylim(0, 1)
    
    ax.set_title(f'{metric_name} Comparison Across Models (JSON Data)')
    ax.set_xticks(np.arange(n_themes))
    ax.set_xticklabels(themes, rotation=45, ha='right')
    ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save the figure
    if output_dir is None:
        output_file = f"json_comparison_{metric_name}.png"
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f"json_comparison_{metric_name}.png")
        
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {metric_name} JSON comparison to '{output_file}'")
    plt.close()


if __name__ == "__main__":
    main()