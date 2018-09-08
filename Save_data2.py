import numpy as np
import matplotlib.pyplot as plt
import os

#=================================================Extracting Results====================================
game_name = 'Frostbite'
plot_mode = 'mean'
config_list = ['naivebackward','theanobackward1','theanobackward2' ,'theanobackward3']
Seed =[12,123,1234,12345,23,234,2345,23456]
epoch = 40
ymin = 0
ymax = 2500

	
#backward cumul data processing========================================================================================================================================
Backward_Test = np.zeros((len(config_list), len(Seed), epoch))
Backward_minTestCount = epoch * np.ones(len(config_list));

config_count = 0
for config in config_list:
  seed_count = 0
  for seed in Seed:
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
	


plt.axis([0, 10, ymin, ymax])
#plt.xticks([], [])
plt.xlabel('Million Frames')
plt.ylabel('Score')
plt.title(game_name + " cumulative test score, average of " +str(len(Seed)) +" random seeds")
plt.legend(['tensorflow','train test no op','test no op','no no op'],loc = 'upper left')
plt.savefig("Plots/" +game_name + "_test_cumul3.png", dpi=160)
plt.close()  
  
  
#backward data processing========================================================================================================================================
Backward_Test = np.zeros((len(config_list), len(Seed), epoch))
Backward_minTestCount = epoch * np.ones(len(config_list));

config_count = 0
for config in config_list:
  seed_count = 0
  for seed in Seed:
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
plt.legend(['tensorflow','train test no op','test no op','no no op'],loc = 'upper left')
plt.savefig("Plots/" +game_name + "_test3.png", dpi=160)
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
  for seed in Seed:
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
	

plt.axis([0, 10, ymin, ymax])
#plt.xticks([], [])
plt.xlabel('Million Frames')
plt.ylabel('Score')
plt.title(game_name + " moving average test score, average of " +str(len(Seed)) +" random seeds")
plt.legend(['tensorflow','train test no op','test no op','no no op'],loc = 'upper left')
plt.savefig("Plots/" + game_name + "_test_movingaverage3.png", dpi=160)
plt.close()  
 


 
 
 
 
 
 
 
 
 
 
 
 
 
#backward cumul data processing========================================================================================================================================
Backward_Train = np.zeros((len(config_list), len(Seed), epoch))
Backward_minTrainCount = epoch * np.ones(len(config_list));

config_count = 0
for config in config_list:
  seed_count = 0
  for seed in Seed:
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


plt.axis([0, 10, ymin, ymax])
#plt.xticks([], [])
plt.xlabel('Million Frames')
plt.ylabel('Score')
plt.title(game_name + " cumulative Train score, average of " +str(len(Seed)) +" random seeds")
plt.legend(['tensorflow','train test no op','test no op','no no op'],loc = 'upper left')
plt.savefig("Plots/" +game_name + "_Train_cumul3.png", dpi=160)
plt.close()  
  
  
#backward data processing========================================================================================================================================
Backward_Train = np.zeros((len(config_list), len(Seed), epoch))
Backward_minTrainCount = epoch * np.ones(len(config_list));

config_count = 0
for config in config_list:
  seed_count = 0
  for seed in Seed:
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
plt.legend(['tensorflow','train test no op','test no op','no no op'],loc = 'upper left')
plt.savefig("Plots/" +game_name + "_Train3.png", dpi=160)
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
  for seed in Seed:
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


plt.axis([0, 10, ymin, ymax])
#plt.xticks([], [])
plt.xlabel('Million Frames')
plt.ylabel('Score')
plt.title(game_name + " moving average Train score, average of " +str(len(Seed)) +" random seeds")
plt.legend(['tensorflow','train test no op','test no op','no no op'],loc = 'upper left')
plt.savefig("Plots/" + game_name + "_Train_movingaverage3.png", dpi=160)
plt.close()  
 
 




