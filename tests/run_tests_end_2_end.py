### Looking at subset of activations with DeepScan

#This notebook shows the code steps to:

#1. Select hyperparameters to run deepscan.
#2. Load $H_0$ and $H_1$ activations for ALL dataset pairs.
#3. Extract activations across several layers.
#4. Run DeepScan and save output files and metrics.
#5. Visualize results with UpSet plots and Venn diagrams organized by category.

from deepscan.util.sampler import Sampler
from deepscan.util.pvalranges_calculator import PvalueCalculator
from deepscan.util.utils import scan_write_metrics, customsort
from tqdm import tqdm
from multiprocessing import Pool, set_start_method
import numpy as np
from pathlib import Path
import re
import sys
import warnings

warnings.filterwarnings("ignore")

# Set multiprocessing method
try:
    set_start_method('fork')
except RuntimeError:
    # Already set, ignore
    pass

def process_dataset(dataset, data_dir, output_dir, typerun, scoring, size, number_runs, runs):
    """Process a single dataset"""
    
    PATH_NEG = data_dir / f"llama3_negative_31_{dataset}.npy"
    PATH_POS = data_dir / f"llama3_positive_31_{dataset}.npy"
    
    # Delete existing output files to ensure fresh start
    clean_output_file = output_dir / f"clean_output_{dataset}_{scoring}.txt"
    adv_output_file = output_dir / f"adv_output_{dataset}_{scoring}.txt"
    
    if clean_output_file.exists():
        clean_output_file.unlink()
        print(f"Deleted existing file: {clean_output_file}")
    
    if adv_output_file.exists():
        adv_output_file.unlink()
        print(f"Deleted existing file: {adv_output_file}")
    
    # Check if both files exist
    if not PATH_NEG.exists():
        print(f"Missing: {PATH_NEG}")
        return None
    if not PATH_POS.exists():
        print(f"Missing: {PATH_POS}")
        return None
    
    print(f"Loading:")
    print(f"  NEG: {PATH_NEG}")
    print(f"  POS: {PATH_POS}")
    
    try:
        bg = np.load(PATH_NEG, allow_pickle=True)
        abnormal = np.load(PATH_POS, allow_pickle=True)
        print(f"Loaded successfully")
        print(f"  bg shape: {bg.shape}")
        print(f"  abnormal shape: {abnormal.shape}")
    except Exception as e:
        print(f"Error loading files: {e}")
        return None
    
    clean = bg[:size]
    bg = bg[size:]
    
    print(f"After preprocessing:")
    print(f"  bg shape: {bg.shape}, abnormal shape: {abnormal.shape}, clean shape: {clean.shape}\n")
    
    # Run for each key
    for key in ["clean", "abnormal"]:
        print(f"  Run for key: {key}")
        clean_ssize = runs[typerun][key]["clean_ssize"]
        anom_ssize = runs[typerun][key]["anom_ssize"]
        
        if (clean_ssize != 1 and typerun == "group") or (
            clean_ssize == 1 and typerun == "individual"
        ):
            resultsfile = output_dir / f"clean_output_{dataset}_{scoring}.txt"
        if anom_ssize != 0:
            resultsfile = output_dir / f"adv_output_{dataset}_{scoring}.txt"
        
        print(f"    Output file: {resultsfile}")
        
        bg_sorted = customsort(bg, conditional=False)
        pvalcalculator = PvalueCalculator(bg_sorted)
        
        records_pvalue_ranges = pvalcalculator.get_pvalue_ranges(clean, pvaltest="1tail")
        anom_records_pvalue_ranges = pvalcalculator.get_pvalue_ranges(
            abnormal, pvaltest="1tail"
        )
        
        run_number_runs = number_runs
        if anom_ssize == 1 and clean_ssize == 0:
            run_number_runs = anom_records_pvalue_ranges.shape[0]
        elif clean_ssize == 1 and anom_ssize == 0:
            run_number_runs = records_pvalue_ranges.shape[0]
        
        samples, _ = Sampler.sample(
            records_pvalue_ranges,
            anom_records_pvalue_ranges,
            clean_ssize,
            anom_ssize,
            run_number_runs,
            conditional=False,
        )
        
        pool = Pool(processes=5)
        calls = []
        
        for r_indx in range(run_number_runs):
            pred_classes = None
            run_sampled_indices = None
            sampled_indices = None
            
            calls.append(
                pool.apply_async(
                    scan_write_metrics,
                    [
                        samples[r_indx],
                        pred_classes,
                        clean_ssize,
                        anom_ssize,
                        str(resultsfile),
                        1,
                        False,
                        None,
                        scoring,
                        -1,
                        run_sampled_indices,
                    ],
                )
            )
        
        print("    Beginning Scanning...")
        for sample in tqdm(calls, desc=f"Processing {dataset} ({key})"):
            sample.get()
        
        pool.close()
        pool.join()
        print(f"    Completed\n")
    
    return dataset


def categorize_datasets(dataset_names, output_dir, scoring):
    """Categorize datasets into ethics and politics categories"""
    
    # Define category keywords
    ethics_keywords = {
        'cultural-relativism': 'RELAT',      # cultural relativism
        'deontology': 'DEONT',               # deontology
        'moral-nihilism': 'NIHIL',           # moral nihilism
        'utilitarianism': 'UTILI',           # utilitarianism
        'virtue-ethics': 'VIRTU',            # virtue ethics
    }

    politics_keywords = {
        'politically-conservative': 'CONSE',  # politically conservative
        'politically-liberal': 'LIBER',       # politically liberal
        'anti-immigration': 'INMIG',          # anti-immigration
        'anti-lgbtq': 'LGBTQ',                # anti-LGBTQ rights (case-insensitive match)
    }

    personality_keywords = {
        'agreeableness': 'AGREE',            # Big Five: agreeableness
        'conscientiousness': 'CONSC',        # Big Five: conscientiousness
        'extraversion': 'EXTRA',             # Big Five: extraversion
        'neuroticism': 'NEURO',              # Big Five: neuroticism
        'openness': 'OPEN',                  # Big Five: openness
    }

    output_files_dict_ethics = {}
    output_files_dict_politics = {}
    output_files_dict_personality = {}

    for dataset in dataset_names:
        dataset_lower = dataset.lower()
        output_path = output_dir / f"adv_output_{dataset}_{scoring}.txt"

        if not output_path.exists():
            continue

        # Check ethics keywords
        for keyword, label in ethics_keywords.items():
            if keyword in dataset_lower:
                output_files_dict_ethics[label] = str(output_path)
                break

        # Check politics keywords
        for keyword, label in politics_keywords.items():
            if keyword in dataset_lower:
                output_files_dict_politics[label] = str(output_path)
                break

        # Check personality keywords
        for keyword, label in personality_keywords.items():
            if keyword in dataset_lower:
                output_files_dict_personality[label] = str(output_path)
                break

    return output_files_dict_ethics, output_files_dict_politics, output_files_dict_personality


def visualize_categorized_results(output_files_dict_ethics, output_files_dict_politics, output_files_dict_personality, output_dir):
    """Visualize categorized results with UpSet plots and Venn diagrams"""

    try:
        from utils.utils_nodes import most_common_nodes
        from utils.utils_viz import plot_upset_politics, plot_upset_ethical, plot_upset_personality
        import venn
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"WARNING: Visualization modules not available: {e}")
        return
    
    print("\n" + "="*80)
    print("GENERATING CATEGORIZED VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Process Ethics datasets
    if output_files_dict_ethics:
        print("="*80)
        print("ETHICS VISUALIZATION")
        print("="*80 + "\n")
        
        print(f"Visualizing {len(output_files_dict_ethics)} ethics datasets:")
        for label, path in sorted(output_files_dict_ethics.items()):
            print(f"  {label}: {Path(path).name}")
        print()
        
        try:
            ethics_topics_nodes = most_common_nodes(output_files_dict_ethics)
            print(f"Extracted nodes from {len(ethics_topics_nodes)} ethics datasets\n")
            
            # Generate UpSet plot for ethics
            print("Generating UpSet plot for ethics...")
            plot_upset_ethical(ethics_topics_nodes)
            plt.savefig(output_dir / "upset_ethical.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved: upset_ethical.png\n")
            
            # Generate Venn diagram for ethics
            num_ethics = len(ethics_topics_nodes)
            if num_ethics == 5:
                print("Generating Venn diagram (5 sets) for ethics...")
                labels = venn.get_labels(ethics_topics_nodes.values(), fill=["number"])
                fig, ax = venn.venn5(labels, names=ethics_topics_nodes.keys())
                fig.savefig(output_dir / "venn_diagram_ethics_5.png", dpi=300, bbox_inches='tight')
                print("Saved: venn_diagram_ethics_5.png\n")
            elif num_ethics == 4:
                print("Generating Venn diagram (4 sets) for ethics...")
                labels = venn.get_labels(ethics_topics_nodes.values(), fill=["number"])
                fig, ax = venn.venn4(labels, names=ethics_topics_nodes.keys())
                fig.savefig(output_dir / "venn_diagram_ethics_4.png", dpi=300, bbox_inches='tight')
                print("Saved: venn_diagram_ethics_4.png\n")
            elif num_ethics == 3:
                print("Generating Venn diagram (3 sets) for ethics...")
                labels = venn.get_labels(ethics_topics_nodes.values(), fill=["number"])
                fig, ax = venn.venn3(labels, names=ethics_topics_nodes.keys())
                fig.savefig(output_dir / "venn_diagram_ethics_3.png", dpi=300, bbox_inches='tight')
                print("Saved: venn_diagram_ethics_3.png\n")
            
        except Exception as e:
            print(f"ERROR: Error during ethics visualization: {e}\n")
    else:
        print("WARNING: No ethics datasets found\n")
    
    # Process Politics datasets
    if output_files_dict_politics:
        print("="*80)
        print("POLITICS VISUALIZATION")
        print("="*80 + "\n")
        
        print(f"Visualizing {len(output_files_dict_politics)} politics datasets:")
        for label, path in sorted(output_files_dict_politics.items()):
            print(f"  {label}: {Path(path).name}")
        print()
        
        try:
            politics_topics_nodes = most_common_nodes(output_files_dict_politics)
            print(f"Extracted nodes from {len(politics_topics_nodes)} politics datasets\n")
            
            # Generate UpSet plot for politics
            print("Generating UpSet plot for politics...")
            import matplotlib.pyplot as plt
            plot_upset_politics(politics_topics_nodes)
            plt.savefig(output_dir / "upset_politics.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved: upset_politics.png\n")
            
            # Generate Venn diagram for politics
            num_politics = len(politics_topics_nodes)
            if num_politics == 4:
                print("Generating Venn diagram (4 sets) for politics...")
                labels = venn.get_labels(politics_topics_nodes.values(), fill=["number"])
                fig, ax = venn.venn4(labels, names=politics_topics_nodes.keys())
                fig.savefig(output_dir / "venn_diagram_politics_4.png", dpi=300, bbox_inches='tight')
                print("Saved: venn_diagram_politics_4.png\n")
            elif num_politics == 3:
                print("Generating Venn diagram (3 sets) for politics...")
                labels = venn.get_labels(politics_topics_nodes.values(), fill=["number"])
                fig, ax = venn.venn3(labels, names=politics_topics_nodes.keys())
                fig.savefig(output_dir / "venn_diagram_politics_3.png", dpi=300, bbox_inches='tight')
                print("Saved: venn_diagram_politics_3.png\n")
            
        except Exception as e:
            print(f"ERROR: Error during politics visualization: {e}\n")
    else:
        print("WARNING: No politics datasets found\n")

    # Process Personality datasets
    if output_files_dict_personality:
        print("="*80)
        print("PERSONALITY VISUALIZATION")
        print("="*80 + "\n")

        print(f"Visualizing {len(output_files_dict_personality)} personality datasets:")
        for label, path in sorted(output_files_dict_personality.items()):
            print(f"  {label}: {Path(path).name}")
        print()

        try:
            personality_topics_nodes = most_common_nodes(output_files_dict_personality)
            print(f"Extracted nodes from {len(personality_topics_nodes)} personality datasets\n")

            # Generate UpSet plot for personality
            print("Generating UpSet plot for personality...")
            plot_upset_personality(personality_topics_nodes)
            plt.savefig(output_dir / "upset_personality.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved: upset_personality.png\n")

            # Generate Venn diagram for personality
            num_personality = len(personality_topics_nodes)
            if num_personality == 5:
                print("Generating Venn diagram (5 sets) for personality...")
                labels = venn.get_labels(personality_topics_nodes.values(), fill=["number"])
                fig, ax = venn.venn5(labels, names=personality_topics_nodes.keys())
                fig.savefig(output_dir / "venn_diagram_personality_5.png", dpi=300, bbox_inches='tight')
                print("Saved: venn_diagram_personality_5.png\n")
            elif num_personality == 4:
                print("Generating Venn diagram (4 sets) for personality...")
                labels = venn.get_labels(personality_topics_nodes.values(), fill=["number"])
                fig, ax = venn.venn4(labels, names=personality_topics_nodes.keys())
                fig.savefig(output_dir / "venn_diagram_personality_4.png", dpi=300, bbox_inches='tight')
                print("Saved: venn_diagram_personality_4.png\n")
            elif num_personality == 3:
                print("Generating Venn diagram (3 sets) for personality...")
                labels = venn.get_labels(personality_topics_nodes.values(), fill=["number"])
                fig, ax = venn.venn3(labels, names=personality_topics_nodes.keys())
                fig.savefig(output_dir / "venn_diagram_personality_3.png", dpi=300, bbox_inches='tight')
                print("Saved: venn_diagram_personality_3.png\n")

        except Exception as e:
            print(f"ERROR: Error during personality visualization: {e}\n")
    else:
        print("WARNING: No personality datasets found\n")

    print("="*80)
    print("Visualization Summary:")
    print("="*80)
    print(f"Generated categorized visualizations in: {output_dir}")
    print(f"  Ethics:")
    print(f"    - upset_ethical.png - UpSet plot for ethical theories")
    print(f"    - venn_diagram_ethics_*.png - Venn diagram for ethical theories")
    print(f"  Politics:")
    print(f"    - upset_politics.png - UpSet plot for political perspectives")
    print(f"    - venn_diagram_politics_*.png - Venn diagram for political perspectives")
    print(f"  Personality:")
    print(f"    - upset_personality.png - UpSet plot for Big Five personality traits")
    print(f"    - venn_diagram_personality_*.png - Venn diagram for Big Five personality traits")
    print()


def main():
    """Main function"""
    
    # Configuration
    typerun = "group"
    scoring = "bj"
    model = "Meta-Llama-3-8B-Instruct"
    size = 200
    number_runs = 100
    
    # Hyperparameters for different run types
    runs = {
        "group": {
            "clean": {"clean_ssize": 100, "anom_ssize": 0},
            "abnormal": {"clean_ssize": 50, "anom_ssize": 50},
        }
    }
    
    # Find all dataset pairs
    data_dir = Path("../data/deepscanandactivations/npyactivations")
    output_dir = Path("../output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning directory: {data_dir}")
    
    # Get all negative files
    negative_files = sorted(data_dir.glob("llama3_negative_31_*.npy"))
    print(f"Found {len(negative_files)} negative files")
    
    # Extract dataset names from negative files
    dataset_names = set()
    for neg_file in negative_files:
        # Extract dataset name: llama3_negative_31_DATASET_NAME.npy
        match = re.search(r"llama3_negative_31_(.+)\.npy", neg_file.name)
        if match:
            dataset_name = match.group(1)
            dataset_names.add(dataset_name)
    
    dataset_names = sorted(dataset_names)
    print(f"Found datasets: {dataset_names}\n")
    
    # Process each dataset
    processed_datasets = []
    for dataset in dataset_names:
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset}")
        print(f"{'='*80}\n")
        
        result = process_dataset(dataset, data_dir, output_dir, typerun, scoring, size, number_runs, runs)
        if result:
            processed_datasets.append(result)
    
    print("\n" + "="*80)
    print("All datasets processed!")
    print("="*80)
    
    # Evaluate results for all datasets
    from utils.utils_nodes import get_anom_nodes
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80 + "\n")
    
    results_summary = []
    
    for dataset in processed_datasets:
        PATH_OUTPUT_ABN = output_dir / f"adv_output_{dataset}_{scoring}.txt"
        PATH_OUTPUT_CLN = output_dir / f"clean_output_{dataset}_{scoring}.txt"
        
        if PATH_OUTPUT_ABN.exists():
            try:
                _, _, precision, recall, _ = get_anom_nodes(str(PATH_OUTPUT_ABN))
                precision_mean = np.array(precision).mean()
                precision_std = np.array(precision).std()
                recall_mean = np.array(recall).mean()
                recall_std = np.array(recall).std()
                
                results_summary.append({
                    'dataset': dataset,
                    'precision_mean': precision_mean,
                    'precision_std': precision_std,
                    'recall_mean': recall_mean,
                    'recall_std': recall_std
                })
            except Exception as e:
                print(f"WARNING: Error processing {dataset}: {e}")
        else:
            print(f"WARNING: Output file not found: {PATH_OUTPUT_ABN}")
    
    # Display results table
    if results_summary:
        print("\n" + "="*80)
        print("PRECISION AND RECALL SUMMARY TABLE")
        print("="*80 + "\n")
        print(f"{'Dataset':<50} {'Precision':<25} {'Recall':<25}")
        print(f"{'='*50} {'='*25} {'='*25}")
        
        for result in results_summary:
            dataset_name = result['dataset']
            precision_str = f"{result['precision_mean']:.4f} +/- {result['precision_std']:.4f}"
            recall_str = f"{result['recall_mean']:.4f} +/- {result['recall_std']:.4f}"
            print(f"{dataset_name:<50} {precision_str:<25} {recall_str:<25}")
        
        print("\n" + "="*80)
        print(f"Total datasets evaluated: {len(results_summary)}")
        print("="*80 + "\n")
    else:
        print("WARNING: No results to display\n")
    
    # Categorize and visualize
    output_files_dict_ethics, output_files_dict_politics, output_files_dict_personality = categorize_datasets(processed_datasets, output_dir, scoring)
    visualize_categorized_results(output_files_dict_ethics, output_files_dict_politics, output_files_dict_personality, output_dir)


if __name__ == '__main__':
    main()