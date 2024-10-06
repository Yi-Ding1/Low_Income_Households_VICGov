"""
This program goes through all .py files
Authors: Subject Group W08G4
"""

import importlib.util
import sys
import matplotlib.pyplot as plt

# function for executing a .py file
def run_script(script_path):
    spec = importlib.util.spec_from_file_location("module.name", script_path)
    script_module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = script_module
    spec.loader.exec_module(script_module)
    plt.close()

# list of file paths
file_paths = [
    'data_preprocess.py',
    'data_analysis/exploratory_data_analysis/histogram_plot.py',
    'data_analysis/exploratory_data_analysis/statistics_table.py',
    'data_analysis/correlation_analysis/correlation_and_mi.py',
    'machine_learning/linear_regression_model.py',
    'machine_learning/Kmean_scatter_plot.py',
    'machine_learning/Kmean_with_PCA.py',
    'machine_learning/KNN_Classification_Model.py'
]

# run all of the python files
for fpath in file_paths:
    print("-" * 30)
    run_script(fpath)
