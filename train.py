

import argparse
import os
import itertools
from pathlib import Path
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import ogb
import ogb.lsc
from ogb.lsc.pcqm4mv2_pyg import  PygPCQM4Mv2Dataset
from accelerate import Accelerator
from torch_geometric import loader
from ogb.lsc import PCQM4Mv2Evaluator

import sys
from model import GCN


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--root_dir", type=str, default="dataset", help="name of the source dataset")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")

    parser.add_argument("--criterion", type=str, default="MAE", help="Loss function")


    parser.add_argument("--hidden_channels", type=int, default=128, help="hidden layers dimension")
    parser.add_argument("--node_emb", type=int, default=128, help="node embedding dimension")
    parser.add_argument("--edge_emb", type=int, default=128, help="edge embedding dimension")
    parser.add_argument("--n_heads", type=int, default=3, help="Number of heads of each transformer layer")

    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of CPU threads to use during batch generation")
    
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")

    parser.add_argument("--wandb", action="store_true", help="If passed, will log to Weights and Biases.")

    parser.add_argument(
        "--wandb_entity",
        required="--wandb" in sys.argv,
        type=str,
        help="Entity name to push to the wandb logged data, in case args.wandb is specified.",
    )    
    parser.add_argument("--output_dir", type=Path, default=Path("./output"), help="Name of the directory to dump generated images and models separately, during training.")
    return parser.parse_args(args=args)

def main(args):
    accelerator = Accelerator(fp16=args.fp16, cpu=args.cpu, mixed_precision=args.mixed_precision)
    
    if args.wandb and accelerator.is_local_main_process:
        import wandb
        wandb.init(project='PCQM4Mv2',
                       entity=args.wandb_entity)    
        wandb.config = vars(args)

        

    #ROOT = '/home/xrh1/datasets/pcqm4m/'
    dataset =  PygPCQM4Mv2Dataset(root = args.root_dir)
    evaluator = PCQM4Mv2Evaluator()

    split_dict = dataset.get_idx_split()
    train_idx = split_dict['train'] # numpy array storing indices of training molecules
    valid_idx = split_dict['valid'] # numpy array storing indices of validation molecules
    testdev_idx = split_dict['test-dev'] # numpy array storing indices of test-dev molecules
    testchallenge_idx = split_dict['test-challenge'] # numpy array storing indices of test-challenge molec


    model = GCN(hidden_channels=args.hidden_channels, node_emb_dim=args.node_emb, edge_emb_dim=args.edge_emb)
    train_loader = loader.DataLoader(dataset[train_idx], batch_size=args.batch_size, shuffle=True)
    val_loader = loader.DataLoader(dataset[valid_idx], batch_size=args.batch_size)
    test_dev_loader = loader.DataLoader(dataset[testdev_idx], batch_size=args.batch_size)
    test_challenge_loader = loader.DataLoader(dataset[testchallenge_idx], batch_size=args.batch_size)



    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,  betas=(args.beta1, args.beta2))

    model, optimizer,  train_loader, val_loader,test_dev_loader,test_challenge_loader = accelerator.prepare(model, optimizer, train_loader, val_loader,test_dev_loader,test_challenge_loader)

    if args.criterion == 'MAE':
        criterion = torch.nn.L1Loss()
    else:
        criterion = torch.nn.MSELoss()

    def train():
        model.train()

        for iter, data in enumerate(train_loader,1):  # Iterate in batches over the training dataset.

            data.y = data.y.reshape(-1,1)
            #data_batch = data.to(DEVICE)  
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            accelerator.backward(loss)
            #loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
            if accelerator.is_local_main_process:
                wandb.log({'Train {args.criterion} loss: ': loss.item() })
    def test(loader):
        model.eval()

        correct = 0
        
        from collections import defaultdict
        res = defaultdict(float)
        for data in loader:  # Iterate in batches over the training/test dataset.
            #data = data.to(DEVICE)

            y_pred = model(data.x, data.edge_index, data.edge_attr, data.batch)  
            input_dict = {'y_pred': y_pred.flatten(), 'y_true': data.y}
            result_dict = evaluator.eval(input_dict)
            if accelerator.is_local_main_process:
                for k in result_dict:
                    res[k] += result_dict[k]

            #pred = out.argmax(dim=1)  # Use the class with highest probability.
            #correct += float( ( torch.pow((pred - data.y),2).sum() ))  # Check against ground-truth labels.
        if accelerator.is_local_main_process:
            for k in res:
                res[k] = res[k]/len(loader.dataset)
        return res
        #return correct / len(loader.dataset)  # Derive ratio of correct predictions.


    for epoch in range(1, args.num_epochs+1):
        train()
        train_metrics = test(train_loader)
        test_metrics = test(val_loader)
        if accelerator.is_local_main_process:
            print(f'Epoch: {epoch:03d}/{args.num_epochs}')
            if args.wandb :
                test_metrics.update({'Epoch': epoch})
                wandb.log(test_metrics)
                


if __name__ == "__main__":

    args = parse_args()
    main(args)
