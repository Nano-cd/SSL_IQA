import numpy as np


srocc = []
plcc = []
with open('results/WAD_baseline2k.txt', 'r') as infile:
    for line in infile:
        plcc.append(float(line.split()[0]))
        srocc.append(float(line.split()[1]))
print(np.mean(np.array(srocc)), np.median(np.array(srocc)), np.std(np.array(srocc)))
print(np.mean(np.array(plcc)), np.median(np.array(plcc)), np.std(np.array(plcc)))


