import numpy as np
import os 
import csv
filedir= "/home/sichoi/test/Q-Optimality-Tightening/code/breakout/"
filenames = os.listdir("/home/sichoi/test/Q-Optimality-Tightening/code/breakout/")
seaquest_list = []
frostbite_list = []
for filename in filenames:
	filenamedir = filedir+filename
	seaquest_list.append(filenamedir)
#	if 'frostbite' in filename:
#		frostbite_list.append(filenamedir)


#print(pong_list)
#print(breakout_list)
print(seaquest_list)
#print(frostbite_list)
#pong_array = []
#breakout_array = []
seaquest_array = []

for filename in seaquest_list:
        resultpath = filename+'/results.csv'
        f=open(resultpath,'r')
        csvReader= csv.reader(f)
        count=0
        for rows in csvReader:
                if 0<count<41:
                        seaquest_array.append(rows[3])
                count +=1
print(np.size(np.array(seaquest_array)))
seaquest_array = np.array(seaquest_array).astype(float)
#print(seaquest_array)
seaquest_check = np.reshape(seaquest_array,[-1,40])

seaquest_max = np.max(seaquest_check,axis=1)
print(seaquest_max)
print(np.mean(seaquest_max))
