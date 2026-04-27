# ===============================================
# test_finetune.py
# ===============================================

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from scipy import stats

from model_non_cf_swim import UNet_Ablation_non_cf
from dataset_fine import CustomDataset

# -----------------------------------------------
# Dice
# -----------------------------------------------
def compute_dice(pred, mask, threshold=0.5):
    pred_bin = (pred > threshold).float()
    mask_bin = (mask > threshold).float()
    smooth = 1e-6
    intersection = (pred_bin * mask_bin).sum()
    union = pred_bin.sum() + mask_bin.sum()
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.item()

# -----------------------------------------------
# IoU
# -----------------------------------------------
def compute_iou(pred, mask, threshold=0.5):
    pred_bin = (pred > threshold).float()
    mask_bin = (mask > threshold).float()
    smooth = 1e-6
    intersection = (pred_bin * mask_bin).sum()
    union = pred_bin.sum() + mask_bin.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


# -----------------------------------------------
def visualize_sample_concat(img, mask, pred, save_dir, filename, dpi=300):
    os.makedirs(save_dir, exist_ok=True)
    img_norm = img.astype(np.float32) / 255.0
    orig_img = img_norm.copy()

    # GT mask（绿色半透明）
    mask_bin = (mask > 0.5).astype(np.float32)
    overlay_gt = np.zeros_like(img_norm)
    overlay_gt[..., 1] = 1.0
    alpha_gt = 0.3
    gt_overlay = orig_img * (1 - alpha_gt * mask_bin[..., None]) + \
                 overlay_gt * (alpha_gt * mask_bin[..., None])


    pred_bin = (pred > 0.5).astype(np.float32)
    overlay_pred = np.zeros_like(img_norm)
    overlay_pred[..., 0] = 1.0
    alpha_pred = 0.4
    pred_overlay = orig_img * (1 - alpha_pred * pred_bin[..., None]) + \
                   overlay_pred * (alpha_pred * pred_bin[..., None])

    concat_img = np.concatenate([orig_img, gt_overlay, pred_overlay], axis=1)
    plt.figure(figsize=(12, 4))
    plt.imshow(concat_img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename}_concat.png"), dpi=dpi)
    plt.close()


# -----------------------------------------------
def compute_ci(data, confidence=0.95):
    a = np.array(data)
    mean = np.mean(a)
    std  = np.std(a, ddof=1)
    n    = len(a)
    se   = std / np.sqrt(n)
    h    = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, std, mean - h, mean + h


# -----------------------------------------------
def bootstrap_classification_metrics(y_true, y_probs, n_bootstrap=1000, threshold=0.5, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    n = len(y_true)
    metrics_bs = {"AUC": [], "ACC": [], "Sensitivity": [], "Specificity": [], "Precision": [], "F1-score": []}

    for _ in range(n_bootstrap):
        idxs = rng.integers(0, n, n)
        y_true_bs = y_true[idxs]
        y_probs_bs = y_probs[idxs]
        y_pred_bs = (y_probs_bs > threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true_bs, y_pred_bs, labels=[0,1]).ravel()
        metrics_bs["AUC"].append(roc_auc_score(y_true_bs, y_probs_bs))
        metrics_bs["ACC"].append(accuracy_score(y_true_bs, y_pred_bs))
        metrics_bs["Sensitivity"].append(recall_score(y_true_bs, y_pred_bs))
        metrics_bs["Specificity"].append(tn / (tn + fp + 1e-6))
        metrics_bs["Precision"].append(precision_score(y_true_bs, y_pred_bs, zero_division=0))
        metrics_bs["F1-score"].append(f1_score(y_true_bs, y_pred_bs))

    # 汇总
    summary = {}
    for k, v in metrics_bs.items():
        arr = np.array(v)
        mean = arr.mean()
        std  = arr.std(ddof=1)
        ci_lower = np.percentile(arr, 2.5)
        ci_upper = np.percentile(arr, 97.5)
        summary[k] = (mean, std, ci_lower, ci_upper)
    return summary


# -----------------------------------------------
def run_test(model, dataloader, save_vis_dir, device="cuda"):
    model.eval()
    os.makedirs(save_vis_dir, exist_ok=True)
    results = []
    all_labels = []
    all_probs  = []
    all_dices = []
    all_ious  = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            label = batch["label"].to(device)
            filenames = batch["filename"]

            seg_logits, cls_logits, _= model(image)
            cls_probs = torch.softmax(cls_logits, dim=1)[:,1]
            seg_probs = seg_logits


            for i in range(image.size(0)):
                img_np = image[i].permute(1,2,0).cpu().numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
                img_np = (img_np*255).astype(np.uint8)
                mask_np = mask[i,0].cpu().numpy()
                pred_np = seg_probs[i,0].cpu().numpy()

                visualize_sample_concat(img_np, mask_np, pred_np,
                                        save_vis_dir,
                                        filename=f"{batch_idx}_{filenames[i]}",
                                        dpi=300)

                dice = compute_dice(seg_probs[i,0], mask[i,0])
                iou  = compute_iou(seg_probs[i,0], mask[i,0])

                results.append({
                    "filename": filenames[i],
                    "label": label[i].item(),
                    "pred_prob": cls_probs[i].item(),
                    "dice": dice,
                    "iou": iou
                })

                all_labels.append(label[i].item())
                all_probs.append(cls_probs[i].item())
                all_dices.append(dice)
                all_ious.append(iou)


    df = pd.DataFrame(results)
    csv_path = os.path.join(save_vis_dir, "test_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved CSV: {csv_path}")


    metrics_ci = bootstrap_classification_metrics(all_labels, all_probs, n_bootstrap=1000)

    metrics_ci["Mean Dice"] = compute_ci(all_dices)
    metrics_ci["Mean IoU"]  = compute_ci(all_ious)


    txt_path = os.path.join(save_vis_dir, "overall_metrics.txt")
    with open(txt_path, "w") as f:
        f.write("===== Overall Test Metrics (with 95% CI) =====\n\n")
        for k, v in metrics_ci.items():
            f.write(f"{k:12}: {v[0]:.4f} ± {v[1]:.4f} (95% CI: {v[2]:.4f} - {v[3]:.4f})\n")

    print(f"[INFO] Saved metrics TXT: {txt_path}")

    return df, metrics_ci


# -----------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_image_dir = r"data_fine/extra_test2/extra_test2_images_1"
    test_mask_dir  = r"data_fine/extra_test2/extra_test2_masks_1"
    test_label_xls = r"data_fine/extra_test2/extra_test2_labels.xlsx"
    model_path     = r"out/result_ablation_non_cf/ablation_best_loss_non_cf.pth"
    save_vis_dir   = r"out/vis/extra_test5_vis_non_cf_swin"

    transform = T.ToTensor()

    test_dataset = CustomDataset(test_image_dir,
                                 test_mask_dir,
                                 test_label_xls,
                                 transform=transform)

    test_loader = DataLoader(test_dataset,
                             batch_size=8,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)

    # -------- Model
    model = UNet_Ablation_non_cf().to(device)

    ckpt = torch.load(model_path, map_location=device)
    new_ckpt = {}

    #non_swin,non_swin_cf,non_cf
    for k, v in ckpt.items():
        if k.startswith("module."):
            k = k.replace("module.", "", 1)

        if k.startswith("BN."):
            k = k.replace("BN.", "Bott.", 1)
        elif k.startswith("RS."):
            k = k.replace("RS.", "Bott.", 1)


    # model,non_ssl
    # for k, v in ckpt.items():
    #     if k.startswith("module."):
    #         k = k.replace("module.", "", 1)
    #
    #     if k.startswith("BN."):
    #         k = k.replace("BN.", "RS.", 1)
    #     elif k.startswith("Bott."):
    #         k = k.replace("Bott.", "RS.", 1)


        new_ckpt[k] = v
    msg = model.load_state_dict(new_ckpt, strict=False)
    print("missing keys:", msg.missing_keys)
    print("unexpected keys:", msg.unexpected_keys)
    print(f"[INFO] Loaded model (BN/Bott -> RS adapted): {model_path}")

    # -------- Run Test
    df_results, metrics_ci = run_test(model, test_loader, save_vis_dir, device=device)
