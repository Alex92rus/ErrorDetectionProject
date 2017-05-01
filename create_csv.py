import fce_api as fd
import csv


fce_data = fd.extract_data('fce_train.gold.max.rasp.old_cat.m2')

with open('clc_fce.csv', 'w', newline='') as csv_file:
    field_names = ['sentence']
    dict_writer = csv.DictWriter(csv_file, fieldnames=field_names)
    dict_writer.writeheader()
    for sentence in fce_data:
        dict_writer.writerow({field_names[0]: sentence[0]})
