import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils import gallery
import argparse
import os


def read_video(input_path, img_size):
    cap = cv.VideoCapture(input_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.resize(frame, (img_size, img_size))
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    return frames


def main(args):
    sequences = []
    masks = []
    for video in os.listdir(args.dir):
        if not video.endswith('.mp4'):
            continue
        print(f'Processing {video}')
        video_path = os.path.join(args.dir, video)
        masks_path = os.path.join(args.dir, video.replace('.mp4', '_mask.avi'))
        frames = read_video(video_path, args.img_size)
        mask = read_video(masks_path, args.img_size)

        # convert masks to binary
        red = mask[..., 0]
        mask = np.zeros_like(red)
        mask[red >= 128] = 1

        # split into sequences
        num_sequences = len(frames) // args.seq_len
        for i in range(num_sequences):
            start = i * args.seq_len
            end = start + args.seq_len
            sequences.append(frames[start:end])
            masks.append(mask[start:end])
    sequences = np.array(sequences)
    masks = np.array(masks)
    train_size = int((1 - args.val_split - args.test_split) * len(sequences))
    val_size = int(args.val_split * len(sequences))
    test_size = int(args.test_split * len(sequences))

    dataset_path = os.path.join(args.dest, args.experiment)
    compressed_path = os.path.join(dataset_path, f'{args.experiment}_real_sl{args.seq_len}.npz')
    np.savez_compressed(compressed_path,
                        train_x={'frames': sequences[:train_size],
                                 'masks': masks[:train_size]},
                        valid_x={'frames': sequences[train_size:train_size + val_size],
                                 'masks': masks[train_size:train_size + val_size]},
                        test_x={'frames': sequences[train_size + val_size:],
                                'masks': masks[train_size + val_size:]})

    norm = plt.Normalize(0.0, 1.0)

    gallery_path = os.path.join(dataset_path, f'{args.experiment}_real_sl{args.seq_len}_samples_masks.jpg')
    fig, ax = plt.subplots(figsize=(sequences.shape[1], 10))
    masks = np.expand_dims(masks, -1)
    result = gallery(np.concatenate(masks[:10]), ncols=sequences.shape[1])
    ax.imshow(np.squeeze(result), interpolation='nearest', cmap=cm.Greys_r, norm=norm)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(gallery_path)

    gallery_path = os.path.join(dataset_path, f'{args.experiment}_real_sl{args.seq_len}_samples_frames.jpg')
    fig, ax = plt.subplots(figsize=(sequences.shape[1], 10))
    result = gallery(np.concatenate(sequences[:10] / 255), ncols=sequences.shape[1])
    ax.imshow(np.squeeze(result), interpolation='nearest', cmap=cm.Greys_r, norm=norm)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(gallery_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--dir', type=str, required=True,
                        help='Path to the directory containing videos and masks')
    parser.add_argument('--dest', type=str, default='data/datasets',
                        help='Path to the directory where the dataset will be saved')
    parser.add_argument('--img_size', type=int, default=32, help='Size of the images')
    parser.add_argument('--seq_len', type=int, default=42, help='Number of frames to consider as a sequence')
    parser.add_argument('--val_split', type=float, default=0.1, help='Fraction of the data to use as validation')
    parser.add_argument('--test_split', type=float, default=0.1, help='Fraction of the data to use as validation')

    args = parser.parse_args()
    main(args)
