# import numpy as np
#
#
# srocc = []
# plcc = []
# rmse = []
# krocc = []
# with open('results/p2p_nnid_overall.txt', 'r') as infile:
#     for line in infile:
#         plcc.append(float(line.split()[0]))
#         srocc.append(float(line.split()[1]))
#         rmse.append(float(line.split()[2]))
#         krocc.append(float(line.split()[3]))
# print(np.mean(np.array(srocc)), np.median(np.array(srocc)), np.std(np.array(srocc)))
# print(np.mean(np.array(plcc)), np.median(np.array(plcc)), np.std(np.array(plcc)))
# print(np.mean(np.array(rmse)), np.median(np.array(rmse)), np.std(np.array(rmse)))
# print(np.mean(np.array(krocc)), np.median(np.array(krocc)), np.std(np.array(krocc)))

import numpy as np


D1srocc = []
D1plcc = []
D1rmse = []
D1krocc = []

D2srocc = []
D2plcc = []
D2rmse = []
D2krocc = []

D3srocc = []
D3plcc = []
D3rmse = []
D3krocc = []
with open('results/alex_RKD100_nnid_device.txt', 'r') as infile:
    for line in infile:
        if line.split()[0][1] =='1':
            D1srocc.append(float(line.split()[1]))
            D1plcc.append(float(line.split()[2]))
            D1krocc.append(float(line.split()[3]))
            D1rmse.append(float(line.split()[4]))
        elif line.split()[0][1] =='2':
            D2srocc.append(float(line.split()[1]))
            D2plcc.append(float(line.split()[2]))
            D2krocc.append(float(line.split()[3]))
            D2rmse.append(float(line.split()[4]))
        elif line.split()[0][1] =='3':
            D3srocc.append(float(line.split()[1]))
            D3plcc.append(float(line.split()[2]))
            D3krocc.append(float(line.split()[3]))
            D3rmse.append(float(line.split()[4]))

print(np.mean(np.array(D1srocc)), np.median(np.array(D1srocc)), np.std(np.array(D1srocc)))
print(np.mean(np.array(D1plcc)), np.median(np.array(D1plcc)), np.std(np.array(D1plcc)))
print(np.mean(np.array(D1rmse)), np.median(np.array(D1rmse)), np.std(np.array(D1rmse)))
print(np.mean(np.array(D1krocc)), np.median(np.array(D1krocc)), np.std(np.array(D1krocc)))


print(np.mean(np.array(D2srocc)), np.median(np.array(D2srocc)), np.std(np.array(D2srocc)))
print(np.mean(np.array(D2plcc)), np.median(np.array(D2plcc)), np.std(np.array(D2plcc)))
print(np.mean(np.array(D2rmse)), np.median(np.array(D2rmse)), np.std(np.array(D2rmse)))
print(np.mean(np.array(D2krocc)), np.median(np.array(D2krocc)), np.std(np.array(D2krocc)))


print(np.mean(np.array(D3srocc)), np.median(np.array(D3srocc)), np.std(np.array(D3srocc)))
print(np.mean(np.array(D3plcc)), np.median(np.array(D3plcc)), np.std(np.array(D3plcc)))
print(np.mean(np.array(D3rmse)), np.median(np.array(D3rmse)), np.std(np.array(D3rmse)))
print(np.mean(np.array(D3krocc)), np.median(np.array(D3krocc)), np.std(np.array(D3krocc)))