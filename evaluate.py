import numpy as np
import argparse
import os
from utils import gallery
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2 as cv


def compute_psnr_video(gt, pred):
    return np.mean([cv.PSNR(gt[i], pred[i], 1.0) for i in range(gt.shape[0])])


def main(args):
    gt_data = np.load(os.path.join(args.gt_file, args.experiment, f'{args.experiment}_sl{args.total_len}.npz'),
                      allow_pickle=True)
    gt = gt_data['train_x'].item()['frames'][0]
    gt = gt[-args.test_seq_len:]
    gt = gt / 255.0
    gt = gt.astype(np.float32)

    pred_ppi_path = os.path.join(args.out_dir, args.experiment, 'test_images_ppi.npz')
    pred_paig_path = os.path.join(args.out_dir, args.experiment, 'test_images_paig.npz')

    pred_ppi_data = np.load(pred_ppi_path)['arr_0']
    pred_paig_data = np.load(pred_paig_path)['arr_0']

    print(gt.shape, pred_paig_data.shape, pred_ppi_data.shape)
    assert gt.shape == pred_paig_data.shape == pred_ppi_data.shape, 'Shapes of ground truth and predictions do not match'

    combined = np.stack([gt, pred_paig_data, pred_ppi_data], axis=0)

    norm = plt.Normalize(0.0, 1.0)
    gallery_path = os.path.join(args.out_dir, args.experiment, f'comparison.jpg')
    fig, ax = plt.subplots(figsize=(combined.shape[1], 10))
    result = gallery(np.concatenate(combined), ncols=combined.shape[1])
    ax.imshow(np.squeeze(result), interpolation='nearest', cmap=cm.Greys_r, norm=norm)
    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis='y', which='major', labelsize=20)
    ax.set_yticks([16.7, 50, 83.3])
    ax.set_yticklabels(['GT', 'PAIG', 'PPI'])
    fig.tight_layout()
    fig.savefig(gallery_path)

    psnr_paig = compute_psnr_video(gt, pred_paig_data)
    psnr_ppi = compute_psnr_video(gt, pred_ppi_data)

    print('PSNR comparison (higher is better):')
    print(f'PSNR PAIG: {round(psnr_paig, 2)}')
    print(f'PSNR PPI: {round(psnr_ppi, 2)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--gt_file', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--test_seq_len', type=int, default=30)
    parser.add_argument('--total_len', type=int, default=42)
    args = parser.parse_args()
    main(args)
