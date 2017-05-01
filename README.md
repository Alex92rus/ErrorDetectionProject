This is the source code for the thesis project: Evaluation of Crowdsourcing for Automatic Generalised Language Error Detection:

Project abstract
The project is aimed to create an English language error detection crowdsourcing experiment on the adopted crowdsourcing platform Amazon Mechanical Turk. The quality and representation of the annotations gathered are then evaluated using a variety of existing state-of-the-art techniques as well as new and innovative methods.

Prerequisites
Python 3.5
Libraries:
numpy
pandas
tensorflow
matplotlib
itertools
collections

Usage
The code in this directory was used for the analysis in the thesis: Evaluation of Crowdsourcing for Automatic Generalised Language Error Detection. To use the methods – mainly in agreement.py it is recommended to make a python environment with Virtualenv and run it in an IDE such as Pycharm.






Code organization
Most of functions and modules are implemented in agreement_methods.py file. Besides, there are a series of Python scripts and Jupyter notebooks that implement some necessary scripts. Other modules and dependencies are as follows:

Directories:
agreement_methods - analysis of the agreement methods
batch_input - input of the AMT experiment sentence batch
batch_results - results of the AMT experiment sentence batch
error_counts - counts per error classes
classifier – directory of the MLP classifier, main file mlp_main_classifier
experiment_output_m2 -  output of the experiment in m2 format
experiment_output_tsv - LSTM  experiment output data
fce_data_sets - the fce CLC data set both in LSTM and m2 formats
location_analysis - location analysis
lstm_output - output from the LSTM
old_version - old metric methods
plots - plots
small_experiment - test data experiment
train_data_chunks  - chunked data for classifier
weighting_votes_results - results from the weighting votes method
env - virtual environment
Files:
agreement.py - the main code file for the analysis
convert_m2.py - helper function
create_csv.py - helper file creator functions
fce_api.py - the API for fce data sets
settings.py - configuration properties - file locations
