import re
import pandas as pd
import fce_api
import config
import collections
import operator
import matplotlib.pyplot as plt



fce_data = fce_api.extract_data(config.FILENAME)

#fce_data = fce_data[:(len(fce_data) * 2) // 3]

error_count = collections.defaultdict(int)
for sentence, errors in fce_data:
    tokens = re.split(r'(\s+)', sentence)
    word_window_size = min(len(tokens), config.WINDOWSIZE)
    for i in range(0, len(tokens) - word_window_size + 1):
        window_tuple = (tokens[i:i + word_window_size],)
        window_range = range(round(i / 2), round((i + word_window_size) / 2))
        for error in errors:
            if error[0] in window_range or error[1] in window_range:
                if len(window_tuple) < 2:
                    window_tuple = window_tuple + (error[2],)
                    error_count['error'] += 1
        # if there is no error
        if len(window_tuple) == 1:
            error_count['no_error'] += 1

print(error_count)
sorted_counts = sorted(error_count.items(), key=operator.itemgetter(1))
print(sorted_counts)


total = sum(list(error_count.values()))

for key in error_count.keys():
    percent = (error_count[key]/total) * 100
    print(key + ': ' + str(percent))

series = pd.Series(list(error_count.values()), error_count.keys(), name='Error Types')
series.plot.pie(figsize=(6, 6), autopct='%.2f')
plt.show()