# python3
# -*- coding:utf-8 -*-
"""
适配新版 dataset.py 的回归训练脚本
- 数据文件: synergybliss_processed.txt
- 生成对称样本
- 全局特征标准化 (在 prepare 中完成)
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader as GeometricDataLoader
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index
import torch.nn.functional as F

# 导入模型和数据处理类
from dataset import GetData, MyTestDataset, load_cell_data, CELL_DIR, CELL_FEA_DIR, DATAS_DIR, DATASET_NAME
from model import DSPSCL, device


def plot_regression_results(y_true, y_pred, pcc, scc, fold, save_dir):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min_val, max_val, 100)
    plt.plot(x_line, p(x_line), 'g-', label=f'PCC={pcc:.4f}')

    plt.xlabel('True')
    plt.ylabel('Pred')
    plt.title(f'Fold {fold}')
    plt.legend()
    plt.grid(alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'fold_{fold}.png'), dpi=150)
    plt.close()


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 使用 /8t 分区保存结果，避免根分区空间不足
    if os.path.exists('/8t'):
        results_dir = f'/8t/wyt/results/results_{DATASET_NAME}_regression'
    else:
        results_dir = os.path.join(base_dir, 'results', f'results_{DATASET_NAME}_regression')
    
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    # 1. 加载数据
    df = GetData()
    cell_features1, cell_features2 = load_cell_data(CELL_DIR, CELL_FEA_DIR)
    available_cells = set(cell_features1[:, 0])

    # 准备药物特征 (全局标准化)
    d_feature = df.prepare()
    durg_dataset = df.get_feature(d_feature)

    print(f"\n{'=' * 60}")
    print(f"Dataset: {DATASET_NAME} (REGRESSION)")
    print(f"Total samples: {len(durg_dataset['drug_encoding'])}")
    print(f"{'=' * 60}\n")

    fold_results = []

    # 2. 5-Fold CV
    for fold in range(5):
        print(f"\n{'=' * 30} Fold {fold} {'=' * 30}")

        # 划分数据
        traindata, testdata = df.slipt(durg_dataset, foldnum=fold)
        
        # 过滤无效细胞系
        traindata = traindata[traindata['cellname'].isin(available_cells)].reset_index(drop=True)
        testdata = testdata[testdata['cellname'].isin(available_cells)].reset_index(drop=True)

        print(f"Train samples: {len(traindata)}, Test samples: {len(testdata)}")

        if len(traindata) == 0 or len(testdata) == 0:
            print(f"Skipping fold {fold}: empty dataset")
            continue

        # 创建 PyTorch-Geometric 数据集
        train_dataset = MyTestDataset(
            root=DATAS_DIR,
            dataset=f'train_fold{fold}_{DATASET_NAME}',
            xt=traindata['cellname'].tolist(),
            y=traindata['label'].tolist(),
            xd1=[p[0] for p in traindata['drug_encoding']],
            xd2=[p[1] for p in traindata['drug_encoding']],
            xt_feature1=cell_features1,
            xt_feature2=cell_features2
        )

        test_dataset = MyTestDataset(
            root=DATAS_DIR,
            dataset=f'test_fold{fold}_{DATASET_NAME}',
            xt=testdata['cellname'].tolist(),
            y=testdata['label'].tolist(),
            xd1=[p[0] for p in testdata['drug_encoding']],
            xd2=[p[1] for p in testdata['drug_encoding']],
            xt_feature1=cell_features1,
            xt_feature2=cell_features2
        )

        train_loader = GeometricDataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = GeometricDataLoader(test_dataset, batch_size=512, shuffle=False)

        # Per-fold 标签归一化 (只用训练集计算)
        y_train_all = torch.cat([data.y for data in train_dataset], dim=0)
        label_mean = y_train_all.mean()
        label_std = y_train_all.std() + 1e-8

        print(f"[Fold {fold}] label mean={label_mean:.4f}, std={label_std:.4f}")

        # 创建模型
        modeldir = f'Modelscl_{DATASET_NAME}_regression'
        os.makedirs(modeldir, exist_ok=True)

        model = DSPSCL(
            modeldir=modeldir,
            foldnum=fold,
            hiddim=8192,
            mmse=1000,
            task='regression'
        )
        # 同步归一化参数到模型实例
        model.y_mean = label_mean.to(device)
        model.y_std = label_std.to(device)

        optimizer = torch.optim.AdamW(
            model.model.parameters(),
            lr=3e-4,
            weight_decay=5e-3
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True
        )

        best_val = 1e9
        best_model_state = None
        patience = 15
        patience_counter = 0
        
        for epoch in range(100):
            model.model.train()
            train_loss = 0
            train_batches = 0
            
            for data in train_loader:
                data = data.to(device)
                drug1 = data.drug1.view(data.num_graphs, -1)
                drug2 = data.drug2.view(data.num_graphs, -1)
                cell1 = data.cell1.view(data.num_graphs, -1)
                cell2 = data.cell2.view(data.num_graphs, -1)

                optimizer.zero_grad()
                reg_out, _, _, _, _ = model.model(drug1, drug2, cell1, cell2)
                pred = reg_out.view(-1)

                y = data.y.view(-1).to(device)
                y_norm = (y - label_mean.to(device)) / label_std.to(device)

                loss = F.smooth_l1_loss(pred, y_norm, beta=1.0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1

            # Validation
            model.model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    drug1 = data.drug1.view(data.num_graphs, -1)
                    drug2 = data.drug2.view(data.num_graphs, -1)
                    cell1 = data.cell1.view(data.num_graphs, -1)
                    cell2 = data.cell2.view(data.num_graphs, -1)

                    reg_out, _, _, _, _ = model.model(drug1, drug2, cell1, cell2)
                    pred = reg_out.view(-1)
                    pred_real = pred * label_std.to(device) + label_mean.to(device)
                    preds.append(pred_real.cpu())
                    trues.append(data.y.view(-1).cpu())

            preds = torch.cat(preds).numpy()
            trues = torch.cat(trues).numpy()
            val_mse = mean_squared_error(trues, preds)

            scheduler.step(val_mse)

            if val_mse < best_val:
                best_val = val_mse
                best_model_state = {k: v.clone() for k, v in model.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss/train_batches:.4f} | Val MSE: {val_mse:.4f} | Best: {best_val:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # 加载最佳模型进行最终评估
        if best_model_state is not None:
            model.model.load_state_dict(best_model_state)

        # 最终评估
        model.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                drug1 = data.drug1.view(data.num_graphs, -1)
                drug2 = data.drug2.view(data.num_graphs, -1)
                cell1 = data.cell1.view(data.num_graphs, -1)
                cell2 = data.cell2.view(data.num_graphs, -1)

                reg_out, _, _, _, _ = model.model(drug1, drug2, cell1, cell2)
                pred = reg_out.view(-1)
                pred_real = pred * label_std.to(device) + label_mean.to(device)
                preds.append(pred_real.cpu())
                trues.append(data.y.view(-1).cpu())

        y_pred = torch.cat(preds).numpy()
        y_true = torch.cat(trues).numpy()

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        pcc = pearsonr(y_true, y_pred)[0]
        scc = spearmanr(y_true, y_pred)[0]
        ci = concordance_index(y_true, y_pred)

        plot_regression_results(y_true, y_pred, pcc, scc, fold, figures_dir)

        print(f"\nFold {fold} Results:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  PCC:  {pcc:.4f}")
        print(f"  SCC:  {scc:.4f}")
        print(f"  CI:   {ci:.4f}")

        fold_results.append({
            'Fold': fold,
            'MSE': mse,
            'RMSE': rmse,
            'PCC': pcc,
            'SCC': scc,
            'CI': ci
        })

    # 汇总结果
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(os.path.join(results_dir, 'fold_results.csv'), index=False)
    
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS (5-Fold CV)")
    print(results_df.to_string(index=False))
    print(f"\n{'=' * 60}")
    print("AVERAGE RESULTS:")
    print(f"  MSE:  {results_df['MSE'].mean():.4f} ± {results_df['MSE'].std():.4f}")
    print(f"  RMSE: {results_df['RMSE'].mean():.4f} ± {results_df['RMSE'].std():.4f}")
    print(f"  PCC:  {results_df['PCC'].mean():.4f} ± {results_df['PCC'].std():.4f}")
    print(f"  SCC:  {results_df['SCC'].mean():.4f} ± {results_df['SCC'].std():.4f}")
    print(f"  CI:   {results_df['CI'].mean():.4f} ± {results_df['CI'].std():.4f}")
    print(f"{'=' * 60}")
