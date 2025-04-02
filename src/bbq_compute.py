import json
from collections import defaultdict


input_file = 'log/bbq/bbq_fewshot_llama3_8b_awq_2k_lora.json'
# Load the JSON file (adjust the file path as needed)
with open(input_file, "r") as f:
    data = json.load(f)

# Assuming the JSON has one top-level key (a timestamp) that contains the results
timestamp = list(data.keys())[0]
results = data[timestamp]

# Prepare a dictionary to accumulate metrics per category
aggregates = defaultdict(lambda: defaultdict(list))

# Loop through keys and group by category if the key follows the pattern "<category>_fold_<number>"
for key, metrics in results.items():
    if "_fold_" in key:
        # Extract the category name (e.g., "age" from "age_fold_0")
        category = key.split("_fold_")[0]
        # For each metric in the fold, store its value in the corresponding category
        for metric, value in metrics.items():
            aggregates[category][metric].append(value)

# Compute the average for each metric per category
averages = {}
for category, metric_dict in aggregates.items():
    for metric, values in metric_dict.items():
        avg_val = sum(values) / len(values)
        averages[f"{category}_{metric}"] = avg_val

# Print the per-category average metrics
print("Averages per Category:")
#print(averages)
for metric_name, avg in averages.items():
    print(f"{metric_name}: {avg}")

# Selected metrics for overall computation
selected_metrics = ['acc_ambiguous', 'acc_disambiguated', 's_ambiguous', 's_disambiguated']
# Prepare a dictionary to accumulate overall values for the selected metrics across all categories
overall_metrics = defaultdict(list)

# Loop through each category and extract the values for the selected metrics
for category, metric_dict in aggregates.items():
    for metric in selected_metrics:
        if metric in metric_dict:
            overall_metrics[metric].extend(metric_dict[metric])

# Compute overall averages for the selected metrics
overall_averages = {}
for metric in selected_metrics:
    values = overall_metrics[metric]
    if values:
        overall_averages[metric] = sum(values) / len(values)
    else:
        overall_averages[metric] = None

print("\nOverall averages across categories for selected metrics:")
#print(overall_averages)
for metric, avg in overall_averages.items():
    print(f"{metric}: {avg}")
