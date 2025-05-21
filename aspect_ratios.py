'''
Plot the aspect ratios of the detected crystals
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# path to file
file_path = "/home/stejan/snow_crystal_segmentation/scs_out/ltu22/particles.txt"

# create dataframe
data = pd.read_csv(file_path, header=0, delim_whitespace=True)

# find the "no_particle" data and delete it
data.drop(data[data.iloc[:,1] == "no_particle"].index, inplace=True)
data["aspect_ratio"] = data["aspect_ratio"].astype(float)

# # plot the data
# plt.scatter(data.index, data["aed"], color='black', s=1)
# plt.title("Aspect ratios")
# plt.ylabel("ar")
# plt.show()

plt.figure()
plt.hist(data["max_dia"], bins=100, edgecolor="black", label='max_dia')
plt.hist(data["min_dia"], bins=100, edgecolor="black", alpha=0.5, label='min_dia')
#plt.hist(data["aed"], bins=100, edgecolor="black", alpha=0.3, label='aed')
plt.title("LTU_22")
plt.legend()
plt.xscale('log')
plt.show()