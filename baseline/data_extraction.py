import os
import importlib
extract_dataset_from_corpus = importlib.import_module("gui-tools", "extract_dataset_from_corpus")

def extract_data():
    real_dataset_stem = "./KuiSCIMA/KuiSCIMA/optical_symbolic_dataset"
    output_total = "./TotalDataset"
    if not os.path.exists(output_total):
        os.mkdir(output_total)
        extract_dataset_from_corpus(real_dataset_stem, output_total)
