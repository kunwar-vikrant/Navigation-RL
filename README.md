# Let's Go Bananas!!

<p align="center">
  <img width="640" height="480" src="https://media3.giphy.com/media/gjaKRybnsBP5rcfMqq/giphy.gif">
</p>

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

<p align="left">
  <img width="420" height="300" src = "https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png">
</p>

## File Structure

- Banana folder contains the necessary unity environment for training and testing.
- dqn_agent.py : Code for the policy followed by the agent.
- model.py : Model architecture of the deep neural networks.
- Navigation.ipynb - Contains the necessary code for training and testing.

### Deep Q-networks

<p align="center">
  <img width="640" height="480" src="https://video.udacity-data.com/topher/2018/May/5aef2add_dqn/dqn.png">
</p>

<p align="left">
  <img width="640" height="480" src = "https://miro.medium.com/max/1400/1*8coZ4g_pRtfyoHmsuzMH6g.png">
</p>


