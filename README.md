## <div>System Requirement</div>
windows : tensorflow=2.13.0  
ubuntu 18.04 : tensorflow=2.13.1

## <div>Install</div>
First git clone 
```bash
git clone https://github.com/gordanLiang/highway-environment-NN-model-training.git
```
cd to file
```bash
cd highway-environment-NN-model-training
```
Using conda environment python==3.8  
```bash
conda create -n <env name> python=3.8
```
activate environment
```bash
conda activate <env name>
```
intstall requirements with 
```bash
pip install -r requirements.txt
```

## <div>Training DQN model with stable baseline 3</div>
Training dqn model using 
```bash
python highway_dqn_train.py
``` 
Testing dqn model using 
```bash
python highway_dqn_test.py
```
It will give 10 score each test 10 rounds and collect reward.

## <div>Generating dataset with pre-trained dqn model</div>
Load model and generate dataset in highway environment by using
```bash
python dataset_making_each1w.py
```
It will generate total 50000 data for each action 10000 data

## <div>Training NN model using dataset</div>
Before generating dataset, we can use it to train the NN model and test it in highway environment.  
Training the NN model using
```bash
python NN_model_train.py
```
Testing in highway environment
```bash
python NN_model_test.py
```
This is the same way tesing dqn model so can compare.
## <div>Result</div>
DQN 10 reward
```bash
[212, 160, 229, 227, 207, 235, 227, 219, 171, 230]
average:211.7
```
NN model 10 reward
```bash
[229, 227, 222, 214, 189, 241, 212, 194, 222, 198]
average:214.8
```
## <div>References</div>
highway envirenment:[https://github.com/Farama-Foundation/HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv)  
stable baseline 3:[https://github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)



