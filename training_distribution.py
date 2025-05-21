import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import re
from scipy.stats import gamma
import numpy as np

source_path = Path("/home/stejan/snow_crystal_segmentation/step3_output/training_data/particles_.txt")
with open(source_path, 'r') as file:
    lines = file.readlines()

header = re.split(r"\s+", lines[0].strip())
df = [re.split(r"\s+", line.strip()) for line in lines[1:]]
print(header)
data = pd.DataFrame(df, columns=header)        

cols_to_convert = [
    "aspect_ratio", "aed", "min_dia", "max_dim",
    "x", "y", "area_ratio", "particle_area", 
    "particle_area_2", "particle_area_3"
]
for col in cols_to_convert:
    data[col] = pd.to_numeric(data[col], errors="coerce")


n = len(data)
print(f"{n=}")

bins = np.logspace(np.log10(10), np.log10(1000), 50)

    # compute the histogram
data_counts1, data_bin_edges1 = np.histogram(data["max_dim"], bins=bins)

bin_widths = np.diff(data_bin_edges1)
bin_center = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
normalized_counts1 = data_counts1 / bin_widths
    
pdf1 = data_counts1 / (np.sum(data_counts1) * bin_widths )

plt.step(bins[:-1], pdf1)
plt.xticks(bins[1:])
#plt.xaxis().set_major_formatter(plt.ScalarFormatter())
plt.tick_params(axis="x", rotation=45)
plt.xscale("log")
plt.title(f"Training dataset {n=}")
plt.xlabel("size")
plt.ylabel("pdf")
plt.show()
