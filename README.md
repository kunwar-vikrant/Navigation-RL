# Let's Go Bananas!!
<p align="center">
  <src ="https://drive.google.com/file/d/1dTCDmDzV0Mi1CaeykSE_CFrkjnCAcGNN/preview" width="640" height="480">
</p>
  

This repository demonstrates some reinforcement learning methods used to train an agent to navigate and collect yellow banana in a square environment.

## Required files and Dependencies tutorial

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]

## File Structure

- Banana folder contains the necessary unity environment for training and testing.
- dqn_agent.py : Code for the policy followed by the agent.
- model.py : Model architecture of the deep neural networks.
- Navigation.ipynb - Contains the necessary code for training and testing.

## Deep Q-networks
<p align="center">
  <src = "https://video.udacity-data.com/topher/2018/May/5aef2add_dqn/dqn.png">
</p>
  
<p align="center">
  <src =  "https://miro.medium.com/max/1400/1*8coZ4g_pRtfyoHmsuzMH6g.png">
</p>


