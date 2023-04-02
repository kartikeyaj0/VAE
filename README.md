# CMPE260 Assignmnet 1
The goal of this assignment is for you to become familiar with gym environment, 
apply vae deep learning algorithm implementation,
and see how the training/inference looks like in code.

### Goals 
* explore the gym framework for training rl agents.
* apply your knowledge of VAE to learn image generation.
* train generative models to produce sample pixel observation images from gym environments.

### What to submit
* your `train_vae.py`.
* a doc with generated images and answers to questions in activities.

### Environment
[OpenAI's Gym](https://gym.openai.com/) is a framework for training reinforcement 
learning agents. It provides a set of environments and a
standardized interface for interacting with those.   
In this assignment we will use the [CartPole](https://gym.openai.com/envs/CartPole-v1/) environment from gym.

### Installation

#### Using conda (recommended)    
1. [Install Anaconda](https://www.anaconda.com/products/individual)

2. Create the env    
`conda create a1 python=3.8` 

3. Activate the env     
`conda activate a1`    

4. install torch ([steps from pytorch installation guide](https://pytorch.org/)):    
- if you don't have an nvidia gpu or don't want to bother with cuda installation:    
`conda install pytorch torchvision torchaudio cpuonly -c pytorch`    
  
- if you have an nvidia gpu and want to use it:    
[install cuda](https://docs.nvidia.com/cuda/index.html)   
install torch with cuda:   
`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`

5. other dependencies   
`conda install -c conda-forge matplotlib gym opencv pyglet`

#### Using pip
`python3 -m pip install -r requirements.txt`

### Code
`MyVAE.py` - your VAE model   
`train_vae.py` - script to collect pixel observations from gym environments using a random policy and train a vae model     
`sample_vae.py` - samples from a vae trained by train_vae.py    


### Activities

1. Finish the `__init__()` in `MyVAE.py` model.
At this point this is not really a VAE yet, but you should be able 
to train the model. Run `train_vae.py` to train. 
Then, run `sample_vae.py` to generate a few images with your model.   
*Note: you can run `MyVAE.py` to quickly test if your model is working.      
Save two generated images.   
What model components are used in the forward pass and in sampling?    
    At this point, our model is just a simple autoencoder and not a variational one.
    In the forward pass, an input image is passed through the encoder, the output of which is passed to z_simple that represents the latent space, and its output is passed to the decoder network.
    During sampling, if the user has not provided a latent space vector, a random vector of dimensions same as the latent space is sampled from normal distribution, and passed to the decoder network. The decoder network is then responsible for generating the image corresponding to the latent space vector.

2. By default, the model behaves as an autoencoder. Upgrade it to 
VAE by modifying `forward()`, `encode()`, and `reparameterize()` 
in `MyVAE.py`.   
Train and save two generated images.      
Describe the difference between the AE and VAE models.
    The difference between AAE and VAE models is that we try to use a probability distribution in the VAE. In AE, the latent space vector ditribution is not guaranteed to be a standard normal distribution. This can lead to problem when using AE for generating samples as the distribution is not guaranteed to be centred or continuous in nature. The problems are further exacerbated when the latent space dimensions are large.
    A VAE is therefore more suitable as it ensures that the latent space vector distribution is a well-behaved standard normal distribution that can produce more reasonable samples than an AE.
What is the reparametrization trick?
    The reparameterization trick uses the fact that we are able to learn the standard normal distribution for the latent space. We therefore have access to the mean vector and covariance matrix for the distribution. Since it is guaranteed that the axes in the latent space are independent, the covariance matrix is essentially a diagonal matrix, which can be presented as a vector. We can therefore calculate the standard deviation for each axis in the latent space. The reparameterization trick can help us sample from N(mu, var) using N(0, 1) by randomly sampling a vector eplison from N(0, 1), which is multiplied by standdard deviation and the result is added to the mean mu.


3. Update the `train_vae.py` to reset 
the environment after the first 20 observations from each episode.    
Train and save two generated images.   
when does the cartpole environment return done=True?
    The episode ends if any one of the following occurs:
    Termination: Pole Angle is greater than ±12°
    Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    Truncation: Episode length is greater than 500 (200 for v0)

 
4. update the `train_vae.py` train vae on 
observations with a custom angle range. Pick some max and min vales for image observations that
will make generated observations look different from the previous outputs. Don't use states that 
too far from the initialization state, so that the sampling doesn't take too long.    
Train and save two generated images.


5. pick [some other gym environment]((https://gym.openai.com/envs/#classic_control)) 
(environments outside the classical control may require you to install additional libraries) 
and train vae on it.    
Train and save two generated images.


Have fun!