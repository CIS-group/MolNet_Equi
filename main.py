import torch
import math
import numpy as np
from model import initialize_weight, MolNet_Eq_Dipole
from model_org import Test_MolNet
from torch_geometric.loader import DataLoader
import argparse
import pickle
import pandas as pd
from tensorboardX import SummaryWriter
import os
from datetime import datetime

model_map = {"MolNet_Eq": MolNet_Eq_Dipole}
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0", help="GPU index to use")
parser.add_argument('--model', default="MolNet_Eq", choices=model_map.keys())
parser.add_argument("--dataset", default='10071_dipole', help="dataset to use")
parser.add_argument('--epoch', default=500, help="number of iteration")
parser.add_argument('--patience', default=50, help="number of patience")
parser.add_argument('--batch', default=8, help="number of batch")
parser.add_argument('--lr', default=0.001, help="learning rate")
parser.add_argument('--l2', default=0, help="l2 regularization")
parser.add_argument('--T_0', default=10, help="T0 for consine annealing")
parser.add_argument('--split', default=10, help="number of split for CV")
parser.add_argument('--useHs', default=True, help="use Hs for molecules or not")
parser.add_argument('--comment', default="", help="comment for experiment")

args = parser.parse_args()

if __name__ == "__main__":

    def train(loader):
        model.train()
        loss_all = 0

        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            mse = torch.nn.MSELoss()
            output = torch.flatten(torch.norm(model(data), dim=1))
            loss_1 = mse(output, torch.flatten(data.y))
            loss_l2 = 0
            rf_model_kvpair = model.state_dict()
            for key, value in rf_model_kvpair.items():
                if "scalar_embed" in key or "sc_fc" in key:
                    loss_l2 += sum(p.pow(2.0).sum() for p in value.cpu())
            loss = loss_1 + 0.005*loss_l2
            loss.backward()
            loss_all += loss_1.item() * data.num_graphs
            optimizer.step()
        return loss_all / len(loader.dataset)

    def test(loader):
        model.eval()
        with torch.no_grad():
            mse_error = 0
            mae_error = 0
            for data in loader:
                data = data.to(device)
                predict = torch.flatten(torch.norm(model(data), dim=1))
                mse = torch.nn.MSELoss()
                mae = torch.nn.L1Loss()
                mse_error += (mse(predict, torch.flatten(data.y)) * data.num_graphs).cpu()# MAE
                mae_error += (mae(predict, torch.flatten(data.y)) * data.num_graphs).cpu()# MAE                
            return mse_error / len(loader.dataset), mae_error / len(loader.dataset)
    hyper = {}
    hyper["model"] = args.model
    hyper["epoch"] = args.epoch
    hyper["patience"] = args.patience
    hyper["lr"] = args.lr
    hyper["batch"] = args.batch
    hyper["l2"] = args.l2
    hyper["split"] = args.split
    hyper["dataset"] = args.dataset
    hyper["T_0"] = args.T_0
    hyper["comment"] = args.comment
    hyper["useHs"] = args.useHs
    results = []
    if hyper["useHs"]:
        dataset = torch.load("./data/{}_useHs.pt".format(hyper["dataset"]))
        print("dataset", "{}_useHs.pt".format(hyper["dataset"]))
    else:
        dataset = torch.load("./data/{}.pt".format(hyper["dataset"]))        
        print("dataset", "{}.pt".format(hyper["dataset"]))
    device = torch.device('cuda:{}'.format(args.gpu))

    now = datetime.now()
    date = now.strftime("%y%m%d%H%M")
    os.mkdir("./result/{}".format(date))

    df =pd.DataFrame(columns=["train_mse", "train_mae", "val_mse", "val_mae"])
    total_train_loss = []
    total_train_mae = []
    total_val_loss = []
    total_val_mae = []
    idx_list = np.array(range(9872))
    np.random.seed(100)
    np.random.shuffle(idx_list)
    for i in range(hyper["split"]):
        os.mkdir("./result/{}/fold{}".format(date,i))
        writer = SummaryWriter(log_dir=("./result/{}/fold{}".format(date,i)))
        fold_result = None
        best_val_loss = 99999
        best_val_mae = None
        checker = 0
        train_loss, train_mae = None, None

        model = model_map[args.model](useHs=hyper["useHs"])
        model.apply(initialize_weight)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hyper["lr"], weight_decay=hyper["l2"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = hyper["T_0"], T_mult=2, eta_min=0)
        len_data = int(9872/10)
        train_idx, val_idx = np.append(idx_list[:len_data*i], idx_list[len_data*(i+1):]), idx_list[len_data*i:len_data*(i+1)]

        train_dataset, val_dataset = [dataset[temp] for temp in train_idx], [dataset[temp] for temp in val_idx]

        val_loader = DataLoader(val_dataset, batch_size=hyper["batch"], shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=hyper["batch"], shuffle=True)

        for epoch in range(1, hyper["epoch"]+ 1):
            lr = scheduler.optimizer.param_groups[0]["lr"]
            train_loss = train(train_loader)
            val_loss, val_mae = test(val_loader)
            _, train_mae = test(train_loader)
            checker += 1
            scheduler.step()
            writer.add_scalars('loss/mse', {'train': train_loss, 'val': val_loss}, i)
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_val_mae = val_mae
                final_train_loss, final_train_mae = test(train_loader)
                checker = 0
                torch.save(model.state_dict(), "./result/{}/fold{}/best_model.pth".format(date,i))
            if checker == hyper["patience"]:
                df = df.append({"train_mse": final_train_loss, "train_mae": final_train_mae, "val_mse": best_val_loss, "val_mae": best_val_mae}, ignore_index=True)
                total_train_loss += [final_train_loss]
                total_train_mae += [final_train_mae]
                total_val_loss += [best_val_loss]
                total_val_mae += [best_val_mae]
                fold_result = {"train_mse": final_train_loss, "train_mae": final_train_mae, "val_mse": best_val_loss, "val_mae": best_val_mae}
                with open("./result/{}/fold{}/fold{}_result.txt".format(date, i, i), "w") as f:
                    print(fold_result, file=f)
                f.close()
                break
            print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, mae: {:.7f}, Val Loss: {:.7f}, Val mae: {:.7f}'.format(epoch, lr, train_loss, train_mae, val_loss, val_mae))
            if epoch == hyper["epoch"]:
                df = df.append({"train_mse": final_train_loss, "train_mae": final_train_mae, "val_mse": best_val_loss, "val_mae": best_val_mae}, ignore_index=True)
                total_train_loss += [final_train_loss]
                total_train_mae += [final_train_mae]
                total_val_loss += [best_val_loss]
                total_val_mae += [best_val_mae]
                fold_result = {"train_mse": final_train_loss, "train_mae": final_train_mae, "val_mse": best_val_loss, "val_mae": best_val_mae}
                with open("./result/{}/fold{}/fold{}_result.txt".format(date, i, i), "w") as f:
                    print(fold_result, file=f)
                f.close()
    df = df.append({"train_mse": np.mean(np.array(total_train_loss)), "train_mae": np.mean(np.array(total_train_mae)), "val_mse": np.mean(np.array(total_val_loss)), "val_mae": np.mean(np.array(total_val_mae))}, ignore_index=True)
    df.to_csv("./result/{}/result.csv".format(date), index=False)
    with open("./result/{}/hyper.txt".format(date), "w") as f:
        print(hyper, file=f)
