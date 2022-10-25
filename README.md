# ChemGraphormer
A **graph convolutional neural network transformer** model to predict HOMO-LUMO gap for the **OGB Large Scale Challenge**

The project natively supports **distributed training** on single or multi gpu machines, with simultaneous **logging of experiments** by means of **wandb**.

## Setup
Install the base requirements by running  
```bash
pip install -r requirements.txt
```
The project also need pytorch (preferably the gpu version).
To find the version that suits your system, visit the official [pytorch website](https://pytorch.org/get-started/locally/) 
Furthermore, this project supports multi-gpu multi-machine training by means of the **accelerate** library.
To set it up, after the installation of accelerate by means of pip, run 
```
accelerate config
```
This will prompt a set of questions in order to properly set up the multi gpu training.
It is important to answer to the question 'How many processes in total will you use?' with the number of gpus that will be employed during training.
Otherwise the library fails to properly configure the environment for multi gpu training(this holds for accelerate 0.6.2, other versions might be subject to changes).

## Dataset 
As shown on the website of the [OGB challenge](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/), the website can be automatically downloaded by means of the ogb library.
Natively this code will download or search for such data at the location pointed by the path given as --root_dir command line argument.

## How to run training
To reap the benefits of distributed training the following has to be run after `accelerate config` is.
```
accelerate launch train.py  command_line_args
```
All command line arguments that would be passed to train.py are specified as is usual in place of the command_line_args string (e.g.)
```
accelerate launch train.py  --num_epochs 200 --n_heads 3 --node_emb 128
```
## Command line arguments
The following parameters can be set to run model training on a variety of model configurations
### Model parameters
- `hidden_channels` input dimension of the Q,K,V matrices of each TransformerConv module. The output size is set equal to hidden_channels for simplicity, although it must be mentioned that each Q,K,V output is of size n_heads*hidden_channels as per the layer definition.  
- `node_emb`: dimension of the embedding of node features
- `edge_emb`:dimension of the embedding of edge features
- `n_heads`: number of heads of the TransformerConv module employed in the model architecture. It influences the output size of each Q,K,V matrix (e.g. n_heads=2 implies a doubled size of the output each of those matrices)
### Optimization: training loop parameters
- `num_epochs`: Number of full iterations to perform over the whole dataset
- `root_dir`: path to where the dataset is located(or the place to which it will be automatically downloaded)
- `batch_size`: number of graphs to batch together
- `criterion`: MAE or MSE, objective function used to optimize the model parameters 
- `lr`: learning rate
- `beta1`: first order coefficient that weighs the history of past gradients 
- `beta2`: second order coefficient that weighs the history of past gradients
### System parameters
- `num_workers`: number of processes to instantiate for multi-process data loading
- `fp16`: whether to enable fp16 training, faster but less accurate
- `mixed_precision`: balanced tradeoff between fp16 and fp32 only training
- `cpu`: whether to run on cpu(by default training will run as configured with the call to accelerate config)
- `wandb`: whether to store and automatically visualize training logs on wandb
- `wandb_entity`: is the account that is used to store the wandb logs if `wandb` is specified.

