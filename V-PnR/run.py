import os
import sys
import random
import numpy as np
import torch

from dataset import *
from model import *
from torch.utils.data import DataLoader, Subset 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def train(model, device, train_loader, optimizer, criterion, noise_ratio=0):
    model.train()
    train_loss = 0.0
    total_graphs = 0
    true_labels = []
    predictions = []
    designs = [] 
    cluster_idx = []
    util = []
    ar = []

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.hyperedge_index, data.batch)
        loss = criterion(output, data.y.view(-1, 1).float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.num_graphs
        total_graphs += data.num_graphs 
        true_labels.extend(data.y.cpu().numpy())
        predictions.extend(output.detach().cpu().numpy())
        designs.extend(data.design)
        cluster_idx.extend(data.cluster_idx)
        util.extend(data.util)
        ar.extend(data.ar)

    return train_loss / total_graphs, (true_labels, predictions, designs, cluster_idx, util, ar) 

def evaluate(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0.0
    total_graphs = 0
    true_labels = []
    predictions = []
    designs = []
    cluster_idx = []
    util = []
    ar = []

    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data.x, data.hyperedge_index, data.batch)
            loss = criterion(output, data.y.view(-1, 1).float())
        test_loss += loss.item() * data.num_graphs
        total_graphs += data.num_graphs 
        true_labels.extend(data.y.cpu().numpy())
        predictions.extend(output.cpu().numpy())
        designs.extend(data.design)
        cluster_idx.extend(data.cluster_idx)
        util.extend(data.util)
        ar.extend(data.ar)

    average_loss = test_loss / len(test_loader.dataset)
    true_labels = np.stack(true_labels).squeeze()
    predictions = np.stack(predictions).squeeze()
    
    return average_loss, (true_labels, predictions, designs, cluster_idx, util, ar)

def print_metrics(true_labels, predictions):
    mae = mean_absolute_error(true_labels, predictions)
    r2 = r2_score(true_labels, predictions)
    print(f'MAE: {mae:.4f}, R2: {r2:.4f}')

    return r2

if __name__ == '__main__':
    import argparse
    import datetime

    current_time = datetime.datetime.now().strftime("%m%d%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--ams', action='store_true')
    parser.add_argument('--noise', type=float, default=0)
    parser.add_argument('--design', type=str, default=None)
    args = parser.parse_args()

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Define hyperparameters
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    weight_decay = args.decay
    use_amsgrad = args.ams
    noise_ratio = args.noise

    print(f"Hyperparameters:")
    print(f"    batch_size: {batch_size}")
    print(f"    epochs: {epochs}")
    print(f"    lr: {lr}")
    print(f"    weight_decay: {weight_decay}")
    print(f"    use_amsgrad: {use_amsgrad}")
    print()

    # Set paths
    data_dir = 
    exp_dir = 
    train_dir = 
    test_dir = 

    ckpt_dir = os.path.join(exp_dir, f'ckpt_{current_time}')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
   
    num_features = 35
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Available device: {device}")

    # Set dataset
    dataset = ClusterDataset(data_dir, exp_dir, train_dir, test_dir, num_features, pickle_path=os.path.join(data_dir, "pickle_path"))

    train_dataset = [d for d in dataset if d.flag == 'train']
    test_dataset = [d for d in dataset if d.flag == 'test']

    if args.design is None:
        num_bins = 10  
        target = [data.y.item() for data in train_dataset]
        bins = np.linspace(min(target), max(target), num_bins)
        binned_target = pd.cut(target, bins, labels=False, include_lowest=True)

        train_idx, valid_idx = train_test_split(range(len(train_dataset)), test_size=0.2, stratify=binned_target, random_state=args.seed)
        train_subset, valid_subset = Subset(train_dataset, train_idx), Subset(train_dataset, valid_idx)

    else:
        test_indices = []
        train_indices = []
        for i in range(len(dataset)):
            design = dataset[i]['design']
            if design == args.design:
                test_indices.append(i)
            else:
                train_indices.append(i)

        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)
        
        print(f'Trainset: else:{len(train_subset)} | Testset: {args.design}:{len(test_subset)}')
        
    print(f'Trainset: {len(train_subset)} | Validset: {len(valid_subset)} | Testset: {len(test_dataset)}')
        
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=16, collate_fn=custom_collate_fn) 
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=16, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, collate_fn=custom_collate_fn)

    # Initialize model, criterion, and optimizer
    model = Model(num_features, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=use_amsgrad)

    best_r2 = float("-inf") 
    best_results = None

    print(f"Start training...")
    for epoch in range(epochs):
        train_loss, train_preds = train(model, device, train_loader, optimizer, criterion, noise_ratio)
        valid_loss, valid_preds = evaluate(model, device, valid_loader, criterion)
        test_loss, test_preds = evaluate(model, device, test_loader, criterion)
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Test Loss: {test_loss:.4f}")
        print('  Test ', end='')
        r2 = print_metrics(test_preds[0], test_preds[1])
        print('  Valid ', end='')
        _ = print_metrics(valid_preds[0], valid_preds[1])
        print('  Train ', end='')
        _ = print_metrics(train_preds[0], train_preds[1])
        print()

        if best_r2 < r2 :
            best_results = (train_preds, valid_preds, test_preds)

        torch.save(model.state_dict(), f'{ckpt_dir}/epoch_{epoch+1}.pt')

    with open('../exp_dir/train_{current_time}.result', 'w') as f:
        results = best_results[0] #train_preds
        true = results[0]
        pred = results[1]
        design = results[2]
        cluster_id = results[3]
        util = results[4]
        ar = results[5]
        for i in range(len(true)):
            f.write(f'{design[i]},{cluster_id[i]},{util[i]},{ar[i]},{true[i]},{pred[i]}\n')
    
    with open('../exp_dir/valid_{current_time}.result', 'w') as f:
        results = best_results[1] #valid_preds
        true = results[0]
        pred = results[1]
        design = results[2]
        cluster_id = results[3]
        util = results[4]
        ar = results[5]
        for i in range(len(true)):
            f.write(f'{design[i]},{cluster_id[i]},{util[i]},{ar[i]},{true[i]},{pred[i]}\n')
    
    with open('../exp_dir/test_{current_time}.result', 'w') as f:
        results = best_results[2] #test_preds
        true = results[0]
        pred = results[1]
        design = results[2]
        cluster_id = results[3]
        util = results[4]
        ar = results[5]
        for i in range(len(true)):
            f.write(f'{design[i]},{cluster_id[i]},{util[i]},{ar[i]},{true[i]},{pred[i]}\n')








