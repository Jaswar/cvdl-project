from skimage.draw import disk
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils import gallery
import argparse

default_experiment_args = {
    'pendulum': {
        'r': 3.0,
        'length': 10.0,
        'max_theta': 3 * np.pi / 4,
        'mass': 5.0,
    },
    'pendulum_scale': {
        'r': 5.0,
        'length': 10.0,
        'max_theta': np.pi / 4,
        'mass': 5.0,
        'proj_dist': 21.0,
        'focal_length': 20.0,
    },
    'pendulum_intensity': {
        'r': 10.0,
        'length': 10.0,
        'max_theta': np.pi / 4,
        'mass': 5.0,
        'proj_dist': 20.0,
    },
    'bouncing_ball_drop': {
        'r': 3.0,
        'g': 9.81,
        'elasticity': 0.5,
    },
    'ball_throw': {
        'r': 3.0,
        'g': 9.81,
        'max_velocity': 20.0,
        'theta': np.pi / 4,
    }
}


def fill_args(args):
    default = default_experiment_args[args.experiment]
    for key, value in default.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def generate_dataset(args):
    assert args.experiment in default_experiment_args

    dataset_path = os.path.join(args.dest, f'{args.experiment}')
    generator = {'pendulum': generate_pendulum_sequence,
                 'pendulum_scale': generate_pendulum_scale_sequence,
                 'pendulum_intensity': generate_pendulum_intensity_sequence,
                 'bouncing_ball_drop': generate_bouncing_ball_drop_sequence,
                 'ball_throw': generate_ball_throw_sequence}[args.experiment]
    sequences = []
    poss = []
    vels = []
    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size
    for i in range(train_size + val_size + test_size):
        if i % 100 == 0:
            print(f'\rGenerating sequence {i}/{train_size + val_size + test_size}', end='')
        seq, ang, vel = generator(args)
        sequences.append(seq)
        poss.append(ang)
        vels.append(vel)
    print(f'\rGenerating sequence {train_size + val_size + test_size}/{train_size + val_size + test_size}')
    sequences = np.array(sequences, dtype=np.uint8)
    poss = np.array(poss, dtype=np.float32)
    vels = np.array(vels, dtype=np.float32)

    print('Saving dataset')
    compressed_path = os.path.join(dataset_path, f'{args.experiment}_sl{args.seq_len}.npz')
    np.savez_compressed(compressed_path,
                        train_x={'frames': sequences[:train_size],
                                 'pos': poss[:train_size],
                                 'vel': vels[:train_size]},
                        valid_x={'frames': sequences[train_size:train_size + val_size],
                                 'pos': poss[train_size:train_size + val_size],
                                 'vel': vels[train_size:train_size + val_size]},
                        test_x={'frames': sequences[train_size + val_size:],
                                'pos': poss[train_size + val_size:],
                                'vel': vels[train_size + val_size:]})

    result = gallery(np.concatenate(sequences[:10] / 255), ncols=sequences.shape[1])

    gallery_path = os.path.join(dataset_path, f'{args.experiment}_sl{args.seq_len}_samples.jpg')
    norm = plt.Normalize(0.0, 1.0)
    fig, ax = plt.subplots(figsize=(sequences.shape[1], 10))
    ax.imshow(np.squeeze(result), interpolation='nearest', cmap=cm.Greys_r, norm=norm)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(gallery_path)


def generate_pendulum_sequence(args):
    assert args.img_size >= 2 * (args.length + args.r)
    assert args.max_theta <= np.pi

    sequence = []
    theta = np.random.uniform(-args.max_theta, args.max_theta)
    vel = 0
    thetas = []
    velocities = []
    for _ in range(args.seq_len):
        thetas.append(theta)
        velocities.append(vel)
        frame = np.zeros((args.img_size, args.img_size, 3))
        x = args.length * np.sin(theta) + args.img_size // 2
        y = args.length * np.cos(theta) + args.img_size // 2

        rr, cc = disk((y, x), args.r)
        frame[rr, cc, :] = (255, 0, 0)
        frame = frame.astype(np.uint8)

        sequence.append(frame)

        for _ in range(args.ode_steps):
            F = -args.mass * 10 * np.sin(theta)
            vel = vel + args.dt / args.ode_steps * F / args.length
            theta = theta + args.dt / args.ode_steps * vel

    return sequence, thetas, velocities


def generate_pendulum_scale_sequence(args):
    assert args.max_theta <= np.pi
    assert args.proj_dist > args.length + args.r

    sequence = []
    theta = np.random.uniform(-args.max_theta, args.max_theta)
    vel = 0
    thetas = []
    velocities = []
    for _ in range(args.seq_len):
        velocities.append(vel)
        thetas.append(theta)
        frame = np.zeros((args.img_size, args.img_size, 3))

        d = (args.length + args.r) * np.sin(theta)
        radius = args.r * args.focal_length / ((args.proj_dist - d) ** 2 - args.r ** 2) ** 0.5
        radius = min(args.img_size // 2, radius)
        rr, cc = disk((args.img_size // 2, args.img_size // 2), radius)
        frame[rr, cc, :] = (255, 0, 0)
        frame = frame.astype(np.uint8)

        sequence.append(frame)

        for _ in range(args.ode_steps):
            F = -args.mass * 10 * np.sin(theta)
            vel = vel + args.dt / args.ode_steps * F / args.length
            theta = theta + args.dt / args.ode_steps * vel

    return sequence, thetas, velocities


def generate_pendulum_intensity_sequence(args):
    sequence = []
    theta = np.random.uniform(-args.max_theta, args.max_theta)
    vel = 0
    thetas = []
    velocities = []
    for _ in range(args.seq_len):
        velocities.append(vel)
        thetas.append(theta)
        frame = np.zeros((args.img_size, args.img_size, 3))

        d = args.length * np.sin(theta)
        intensity = (args.proj_dist - args.length) ** 2 / (args.proj_dist - d) ** 2
        rr, cc = disk((args.img_size // 2, args.img_size // 2), args.r)
        frame[rr, cc, :] = (int(255 * intensity), 0, 0)
        frame = frame.astype(np.uint8)

        sequence.append(frame)

        for _ in range(args.ode_steps):
            F = -args.mass * 10 * np.sin(theta)
            vel = vel + args.dt / args.ode_steps * F / args.length
            theta = theta + args.dt / args.ode_steps * vel

    return sequence, thetas, velocities

def generate_bouncing_ball_drop_sequence(args):
    sequence = []
    height = np.random.uniform(args.r, args.img_size - args.r) # put ball in image
    velocity = 0

    heights = []
    velocities = []

    for _ in range(args.seq_len):
        heights.append(height)
        velocities.append(velocity)

        frame = np.zeros((args.img_size, args.img_size, 3))

        clamped_height = np.clip(int(height), args.r, args.img_size - args.r)

        rr, cc = disk((args.img_size - int(clamped_height), args.img_size // 2), args.r)
        frame[rr, cc, :] = (255, 0, 0)
        frame = frame.astype(np.uint8)

        sequence.append(frame)
    
        for _ in range(args.ode_steps):
            height += velocity * (args.dt / args.ode_steps)
            velocity -= args.g * (args.dt / args.ode_steps)

            if (height <= args.r): # check for bounce
                height = args.r
                velocity = -velocity * args.elasticity # reverse velocity (changing direction) and simulate energy loss of elasticity

    return sequence, heights, velocities

def generate_ball_throw_sequence(args):
    sequence = []
    initial_velocity = np.random.uniform(0, args.max_velocity)
    theta = args.theta

    velocity_x = initial_velocity * np.cos(theta)
    velocity_y = initial_velocity * np.sin(theta)

    position_x = args.r
    position_y = args.img_size // 2

    positions_x = []
    positions_y = []
    velocities_x = []
    velocities_y = []

    for _ in range(args.seq_len):
        positions_x.append(position_x)
        positions_y.append(position_y)
        velocities_x.append(velocity_x)
        velocities_y.append(velocity_y)

        frame = np.zeros((args.img_size, args.img_size, 3))

        clamped_position_y = np.clip(int(position_y), args.r, args.img_size - args.r)
        clamped_position_x = np.clip(int(position_x), args.r, args.img_size - args.r)

        rr, cc = disk((args.img_size - int(clamped_position_y), int(clamped_position_x)), args.r)
        frame[rr, cc, :] = (255, 0, 0)
        frame = frame.astype(np.uint8)

        sequence.append(frame)
    
        for _ in range(args.ode_steps):
            position_x += velocity_x * (args.dt / args.ode_steps)
            position_y += velocity_y * (args.dt / args.ode_steps)

            if (position_y <= args.r):
                velocity_x = 0
                velocity_y = 0
                position_y = args.r
            else:
                velocity_y -= args.g * (args.dt / args.ode_steps)
    return sequence, positions_x, positions_y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--seq_len', type=int)
    parser.add_argument('--dest', type=str, default='data/datasets')
    parser.add_argument('--train_size', type=int, default=10000)
    parser.add_argument('--val_size', type=int, default=1000)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--img_size', type=int, default=32)

    # PAIG specific arguments, might be useless for other models, we will see
    parser.add_argument('--ode_steps', type=int, default=10)
    parser.add_argument('--dt', type=float, default=0.3)

    # Experiment specific attributes, values defined by default_experiment_args
    parser.add_argument('--length', type=float, default=None)
    parser.add_argument('--mass', type=float, default=None)
    parser.add_argument('--max_theta', type=float, default=None)
    parser.add_argument('--r', type=float, default=None)
    parser.add_argument('--focal_length', type=float, default=None)
    parser.add_argument('--proj_dist', type=float, default=None)
    parser.add_argument('--g', type=float, default=None)
    parser.add_argument('--elasticity', type=float, default=None)
    parser.add_argument('--max_velocity', type=float, default=None)
    parser.add_argument('--theta', type=float, default=None)



    args = parser.parse_args()
    fill_args(args)
    generate_dataset(args)
    print('Done')
