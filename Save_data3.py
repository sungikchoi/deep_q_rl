import numpy as np
import matplotlib.pyplot as plt
import os

#=================================================Extracting Results====================================
game_name = 'Atlantis'
plot_mode = 'mean'
config_list = ['naivebackward', 'theanobackward1']
Seed1 =[12,123,1234,12345,23,234,2345,23456]
Seed2 = range(1,9)
epoch = 40
ymin = 0
ymax = 150000

	
#backward cumul data processing========================================================================================================================================
Backward_Test = np.zeros((len(config_list), len(Seed), epoch))
Backward_minTestCount = epoch * np.ones(len(config_list));

config_count = 0
for config in config_list:
  seed_count = 0
  if config == 'naivebackward':
    tempseed = Seed1
  else:
    tempseed = Seed2
  for seed in tempseed:
    TestCount = 0
    file = open('LogFiles/' + game_name + '/' + game_name.lower() + '_' + config + '_' + str(seed) + '.log')
    for line in file.readlines():
      if ("Average eval" in line):
        word_list = line.split()
        Backward_Test[config_count, seed_count-1, TestCount]=float(word_list[3])
        if TestCount > 0 and Backward_Test[config_count,seed_count-1, TestCount-1]>Backward_Test[config_count,seed_count-1, TestCount]:
          Backward_Test[config_count,seed_count-1, TestCount] = Backward_Test[config_count,seed_count-1, TestCount-1]
        TestCount += 1
    Backward_minTestCount[config_count] = min(TestCount, Backward_minTestCount[config_count])		  		
    file.close()
    seed_count += 1
  config_count += 1	

Backward_Test_Mean = np.zeros((len(config_list), epoch))
Backward_Test_Std = np.zeros((len(config_list), epoch))



for i in range(len(config_list)):
  Backward_Test_Mean[i] = np.mean(Backward_Test[i], axis = 0)
  Backward_Test_Std[i] = np.std(Backward_Test[i], axis = 0)
  
for i in range(len(config_list)):
  for j in range(epoch):
    if j >= Backward_minTestCount[i]:
      Backward_Test_Mean[i,j] = 0
      Backward_Test_Std[i,j] = 0

step = np.arange(1, epoch + 1)
#Test Plot generation
fig = plt.figure(1) 

 

for i in range(len(config_list)):
  lw = 0.5
  if config_list[i] == "naivebackward":
    lw = 1.0
  plt.plot(0.25*np.arange(1,int(Backward_minTestCount[i])+1), Backward_Test_Mean[i,0:int(Backward_minTestCount[i])], linewidth = lw)
  plt.fill_between(0.25*step, np.min(Backward_Test[i],axis=0),np.max(Backward_Test[i],axis=0), alpha = 0.2, edgecolor = 'none')
	
plt.title(game_name + ' Test Score (mean of 4 random seeds)\n' + 'Final epsilon = 0.1,' + ' Test epsilon = 0.05' )

plt.axis([0, 10, ymin, ymax])
#plt.xticks([], [])
plt.xlabel('Million Frames')
plt.ylabel('Score')
plt.title(game_name + " cumulative test score, average of " +str(len(Seed)) +" random seeds")
plt.savefig("Plots/" +game_name + "_test_cumul.png", dpi=160)
plt.close()  
  
  
#backward data processing========================================================================================================================================
Backward_Test = np.zeros((len(config_list), len(Seed), epoch))
Backward_minTestCount = epoch * np.ones(len(config_list));

config_count = 0
for config in config_list:
  seed_count = 0
  if config == 'naivebackward':
    tempseed = Seed1
  else:
    tempseed = Seed2
  for seed in tempseed:
    TestCount = 0
    file = open('LogFiles/' + game_name + '/' + game_name.lower() + '_' + config + '_' + str(seed) + '.log')
    for line in file.readlines():
      if ("Average eval" in line):
        word_list = line.split()
        Backward_Test[config_count, seed_count-1, TestCount]=float(word_list[3])
        TestCount += 1 
    Backward_minTestCount[config_count] = min(TestCount, Backward_minTestCount[config_count])		  			
    file.close()
    seed_count += 1
  config_count += 1	

Backward_Test_Mean = np.zeros((len(config_list), epoch))
Backward_Test_Std = np.zeros((len(config_list), epoch))
  
for i in range(len(config_list)):
  Backward_Test_Mean[i] = np.mean(Backward_Test[i], axis = 0)
  Backward_Test_Std[i] = np.std(Backward_Test[i], axis = 0)
  
for i in range(len(config_list)):
  for j in range(epoch):
    if j >= Backward_minTestCount[i]:
      Backward_Test_Mean[i,j] = 0
      Backward_Test_Std[i,j] = 0


print Backward_minTestCount

step = np.arange(1, epoch + 1)
#Test Plot generation
fig = plt.figure(1)  

for i in range(len(config_list)):
  lw = 0.5
  if config_list[i] == "naivebackward":
    lw = 1.0
  plt.plot(0.25*np.arange(1,int(Backward_minTestCount[i])+1), Backward_Test_Mean[i,0:int(Backward_minTestCount[i])], linewidth = lw)
  plt.fill_between(0.25*step, np.min(Backward_Test[i], axis = 0), np.max(Backward_Test[i], axis = 0), alpha = 0.2, edgecolor = 'none')

plt.axis([0, 10, ymin, ymax])
#plt.xticks([], [])
plt.xlabel('Million Frames')
plt.ylabel('Score')
plt.title(game_name + " test score<min,mean,max>, average of " +str(len(Seed)) +" random seeds")
plt.savefig("Plots/" +game_name + "_test.png", dpi=160)
plt.close()  
  

#Moving Average PLOT=====================================================================================================================  


def movingaverage (values, window):
    tmp = np.array([0]*(window-1))
    values_ = np.append(tmp,values)
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values_, weights, 'valid')
    return sma
	

  
#backward data processing========================================================================================================================================
Backward_Test = np.zeros((len(config_list), len(Seed), epoch))
Backward_minTestCount = epoch * np.ones(len(config_list));

config_count = 0
for config in config_list:
  seed_count = 0
  if config == 'naivebackward':
    tempseed = Seed1
  else:
    tempseed = Seed2
  for seed in tempseed:
    TestCount = 0
    file = open('LogFiles/' + game_name + '/' + game_name.lower() + '_' + config + '_' + str(seed) + '.log')
    for line in file.readlines():
      if ("Average eval" in line):
        word_list = line.split()
        Backward_Test[config_count, seed_count-1, TestCount]=float(word_list[3])
        TestCount += 1

    Backward_minTestCount[config_count] = min(TestCount, Backward_minTestCount[config_count])		  		
    file.close()
    seed_count += 1
  config_count += 1	

Backward_Test_Mean = np.zeros((len(config_list), epoch))

Backward_Test_Std = np.zeros((len(config_list), epoch))

  
for i in range(len(config_list)):
  Backward_Test_Mean[i] = np.mean(Backward_Test[i], axis = 0)
  Backward_Test_Std[i] = np.std(Backward_Test[i], axis = 0)

  
for i in range(len(config_list)):
  for j in range(epoch):
    if j >= Backward_minTestCount[i]:
      Backward_Test_Mean[i,j] = 0
      Backward_Test_Std[i,j] = 0
	  
step = np.arange(1, epoch + 1)
#Test Plot generation
fig = plt.figure(1)  

for i in range(len(config_list)):
  lw = 0.5
  if config_list[i] == "naivebackward":
    lw = 1.0
  plt.plot(0.25*np.arange(1,int(Backward_minTestCount[i])+1), movingaverage(Backward_Test_Mean[i,0:int(Backward_minTestCount[i])],4), linewidth = lw)
  plt.fill_between(0.25*step, movingaverage(Backward_Test_Mean[i] - Backward_Test_Std[i],4), movingaverage(Backward_Test_Mean[i] + Backward_Test_Std[i],4), alpha = 0.2, edgecolor = 'none')
	
plt.title(game_name + ' Test Score (mean of 4 random seeds)\n' + 'Final epsilon = 0.1,' + ' Test epsilon = 0.05' )

plt.axis([0, 10, ymin, ymax])
#plt.xticks([], [])
plt.xlabel('Million Frames')
plt.ylabel('Score')
plt.title(game_name + " moving average test score, average of " +str(len(Seed)) +" random seeds")
plt.savefig("Plots/" + game_name + "_test_movingaverage.png", dpi=160)
plt.close()  
 


 
 
 
 
 
 
 
 
 
 
 
 
 
#backward cumul data processing========================================================================================================================================
Backward_Train = np.zeros((len(config_list), len(Seed), epoch))
Backward_minTrainCount = epoch * np.ones(len(config_list));

config_count = 0
for config in config_list:
  seed_count = 0
  if config == 'naivebackward':
    tempseed = Seed1
  else:
    tempseed = Seed2
  for seed in tempseed:
    TrainCount = 0
    file = open('LogFiles/' + game_name + '/' + game_name.lower() + '_' + config + '_' + str(seed) + '.log')
    for line in file.readlines():
      if ("Average training" in line):
        word_list = line.split()
        Backward_Train[config_count, seed_count-1, TrainCount]=float(word_list[3])
        if TrainCount > 0 and Backward_Train[config_count,seed_count-1, TrainCount-1]>Backward_Train[config_count,seed_count-1, TrainCount]:
          Backward_Train[config_count,seed_count-1, TrainCount] = Backward_Train[config_count,seed_count-1, TrainCount-1]
        TrainCount += 1
    Backward_minTrainCount[config_count] = min(TrainCount, Backward_minTrainCount[config_count])		  		
    file.close()
    seed_count += 1
  config_count += 1	

Backward_Train_Mean = np.zeros((len(config_list), epoch))
Backward_Train_Std = np.zeros((len(config_list), epoch))



for i in range(len(config_list)):
  Backward_Train_Mean[i] = np.mean(Backward_Train[i], axis = 0)
  Backward_Train_Std[i] = np.std(Backward_Train[i], axis = 0)
  
for i in range(len(config_list)):
  for j in range(epoch):
    if j >= Backward_minTrainCount[i]:
      Backward_Train_Mean[i,j] = 0
      Backward_Train_Std[i,j] = 0

step = np.arange(1, epoch + 1)
#Train Plot generation
fig = plt.figure(1) 

 

for i in range(len(config_list)):
  lw = 0.5
  if config_list[i] == "naivebackward":
    lw = 1.0
  plt.plot(0.25*np.arange(1,int(Backward_minTrainCount[i])+1), Backward_Train_Mean[i,0:int(Backward_minTrainCount[i])], linewidth = lw)
  plt.fill_between(0.25*step, np.min(Backward_Train[i],axis=0),np.max(Backward_Train[i],axis=0), alpha = 0.2, edgecolor = 'none')
	
plt.title(game_name + ' Train Score (mean of 4 random seeds)\n' + 'Final epsilon = 0.1,' + ' Train epsilon = 0.05' )

plt.axis([0, 10, ymin, ymax])
#plt.xticks([], [])
plt.xlabel('Million Frames')
plt.ylabel('Score')
plt.title(game_name + " cumulative Train score, average of " +str(len(Seed)) +" random seeds")
plt.savefig("Plots/" +game_name + "_Train_cumul.png", dpi=160)
plt.close()  
  
  
#backward data processing========================================================================================================================================
Backward_Train = np.zeros((len(config_list), len(Seed), epoch))
Backward_minTrainCount = epoch * np.ones(len(config_list));

config_count = 0
for config in config_list:
  seed_count = 0
  if config == 'naivebackward':
    tempseed = Seed1
  else:
    tempseed = Seed2
  for seed in tempseed:
    TrainCount = 0
    file = open('LogFiles/' + game_name + '/' + game_name.lower() + '_' + config + '_' + str(seed) + '.log')
    for line in file.readlines():
      if ("Average training" in line):
        word_list = line.split()
        Backward_Train[config_count, seed_count-1, TrainCount]=float(word_list[3])
        TrainCount += 1 
    Backward_minTrainCount[config_count] = min(TrainCount, Backward_minTrainCount[config_count])		  			
    file.close()
    seed_count += 1
  config_count += 1	

Backward_Train_Mean = np.zeros((len(config_list), epoch))
Backward_Train_Std = np.zeros((len(config_list), epoch))
  
for i in range(len(config_list)):
  Backward_Train_Mean[i] = np.mean(Backward_Train[i], axis = 0)
  Backward_Train_Std[i] = np.std(Backward_Train[i], axis = 0)
  
for i in range(len(config_list)):
  for j in range(epoch):
    if j >= Backward_minTrainCount[i]:
      Backward_Train_Mean[i,j] = 0
      Backward_Train_Std[i,j] = 0


print Backward_minTrainCount

step = np.arange(1, epoch + 1)
#Train Plot generation
fig = plt.figure(1)  

for i in range(len(config_list)):
  lw = 0.5
  if config_list[i] == "naivebackward":
    lw = 1.0
  plt.plot(0.25*np.arange(1,int(Backward_minTrainCount[i])+1), Backward_Train_Mean[i,0:int(Backward_minTrainCount[i])], linewidth = lw)
  plt.fill_between(0.25*step, np.min(Backward_Train[i], axis = 0), np.max(Backward_Train[i], axis = 0), alpha = 0.2, edgecolor = 'none')

plt.axis([0, 10, ymin, ymax])
#plt.xticks([], [])
plt.xlabel('Million Frames')
plt.ylabel('Score')
plt.title(game_name + " Train score<min,mean,max>, average of " +str(len(Seed)) +" random seeds")
plt.savefig("Plots/" +game_name + "_Train.png", dpi=160)
plt.close()  
  

#Moving Average PLOT=====================================================================================================================  


def movingaverage (values, window):
    tmp = np.array([0]*(window-1))
    values_ = np.append(tmp,values)
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values_, weights, 'valid')
    return sma
	

  
#backward data processing========================================================================================================================================
Backward_Train = np.zeros((len(config_list), len(Seed), epoch))
Backward_minTrainCount = epoch * np.ones(len(config_list));

config_count = 0
for config in config_list:
  seed_count = 0
  if config == 'naivebackward':
    tempseed = Seed1
  else:
    tempseed = Seed2
  for seed in tempseed:
    TrainCount = 0
    file = open('LogFiles/' + game_name + '/' + game_name.lower() + '_' + config + '_' + str(seed) + '.log')
    for line in file.readlines():
      if ("Average training" in line):
        word_list = line.split()
        Backward_Train[config_count, seed_count-1, TrainCount]=float(word_list[3])
        TrainCount += 1

    Backward_minTrainCount[config_count] = min(TrainCount, Backward_minTrainCount[config_count])		  		
    file.close()
    seed_count += 1
  config_count += 1	

Backward_Train_Mean = np.zeros((len(config_list), epoch))

Backward_Train_Std = np.zeros((len(config_list), epoch))

  
for i in range(len(config_list)):
  Backward_Train_Mean[i] = np.mean(Backward_Train[i], axis = 0)
  Backward_Train_Std[i] = np.std(Backward_Train[i], axis = 0)

  
for i in range(len(config_list)):
  for j in range(epoch):
    if j >= Backward_minTrainCount[i]:
      Backward_Train_Mean[i,j] = 0
      Backward_Train_Std[i,j] = 0
	  
step = np.arange(1, epoch + 1)
#Train Plot generation
fig = plt.figure(1)  

for i in range(len(config_list)):
  lw = 0.5
  if config_list[i] == "naivebackward":
    lw = 1.0
  plt.plot(0.25*np.arange(1,int(Backward_minTrainCount[i])+1), movingaverage(Backward_Train_Mean[i,0:int(Backward_minTrainCount[i])],4), linewidth = lw)
  plt.fill_between(0.25*step, movingaverage(Backward_Train_Mean[i] - Backward_Train_Std[i],4), movingaverage(Backward_Train_Mean[i] + Backward_Train_Std[i],4), alpha = 0.2, edgecolor = 'none')
	
plt.title(game_name + ' Train Score (mean of 4 random seeds)\n' + 'Final epsilon = 0.1,' + ' Train epsilon = 0.05' )

plt.axis([0, 10, ymin, ymax])
#plt.xticks([], [])
plt.xlabel('Million Frames')
plt.ylabel('Score')
plt.title(game_name + " moving average Train score, average of " +str(len(Seed)) +" random seeds")
plt.savefig("Plots/" + game_name + "_Train_movingaverage.png", dpi=160)
plt.close()  
 
 




