from skimage.draw import disk
from skimage.transform import resize
import numpy as np


def generate_pendulum_dataset(dest,
                              train_set_size,
                              valid_set_size,
                              test_set_size,
                              seq_len,
                              img_size,
                              radius=3,
                              length=10,
                              max_theta=3*np.pi/4,
                              ode_steps=10,
                              dt=0.3,
                              mass=5):


    def generate_sequence():
        sequence = []
        theta = np.random.uniform(-max_theta, max_theta)
        vel = 0
        thetas = []
        velocities = []
        for _ in range(seq_len):
            thetas.append(theta)
            velocities.append(vel)
            frame = np.zeros((img_size, img_size, 3))
            x = length * np.sin(theta) + img_size // 2
            y = length * np.cos(theta) + img_size // 2

            rr, cc = disk((y, x), radius)
            frame[rr, cc, :] = (255, 0, 0)
            frame = frame.astype(np.uint8)

            sequence.append(frame)

            for _ in range(ode_steps):
                F = -mass * 10 * np.sin(theta)
                vel = vel + dt / ode_steps * F / length
                theta = theta + dt / ode_steps * vel

        return sequence, thetas, velocities

    sequences = []
    poss = []
    vels = []
    for i in range(train_set_size + valid_set_size + test_set_size):
        if i % 100 == 0:
            print(i)
        seq, ang, vel = generate_sequence()
        sequences.append(seq)
        poss.append(ang)
        vels.append(vel)
    sequences = np.array(sequences, dtype=np.uint8)
    poss = np.array(poss, dtype=np.float32)
    vels = np.array(vels, dtype=np.float32)

    np.savez_compressed(dest,
                        train_x={'frames': sequences[:train_set_size],
                                 'pos': poss[:train_set_size],
                                 'vel': vels[:train_set_size]},
                        valid_x={'frames': sequences[train_set_size:train_set_size+valid_set_size],
                                 'pos': poss[train_set_size:train_set_size+valid_set_size],
                                 'vel': vels[train_set_size:train_set_size+valid_set_size]},
                        test_x={'frames': sequences[train_set_size+valid_set_size:],
                                'pos': poss[train_set_size+valid_set_size:],
                                'vel': vels[train_set_size+valid_set_size:]})

    result = gallery(np.concatenate(sequences[:10] / 255), ncols=sequences.shape[1])

    norm = plt.Normalize(0.0, 1.0)
    fig, ax = plt.subplots(figsize=(sequences.shape[1], 10))
    ax.imshow(np.squeeze(result), interpolation='nearest', cmap=cm.Greys_r, norm=norm)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(dest.split(".")[0] + "_samples.jpg")

def generate_pendulum_scale_dataset(dest,
                              train_set_size,
                              valid_set_size,
                              test_set_size,
                              seq_len,
                              img_size,
                              r=5,
                              length=10,
                              focal_length=20,
                              proj_dist=21,
                              max_theta=np.pi/4,
                              ode_steps=5,
                              dt=0.3,
                              mass=5):
    assert proj_dist >= length + 2 * r

    def generate_sequence():
        sequence = []
        theta = np.random.uniform(-max_theta, max_theta)
        vel = 0
        thetas = []
        velocities = []
        for _ in range(seq_len):
            velocities.append(vel)
            thetas.append(theta)
            frame = np.zeros((img_size, img_size, 3))

            d = (length + r) * np.sin(theta)
            radius = r * focal_length / ((proj_dist - d) ** 2 - r ** 2) ** 0.5
            radius = min(img_size // 2, radius)
            rr, cc = disk((img_size // 2, img_size // 2), radius)
            frame[rr, cc, :] = (255, 0, 0)
            frame = frame.astype(np.uint8)

            sequence.append(frame)

            for _ in range(ode_steps):
                F = -mass * 10 * np.sin(theta)
                vel = vel + dt / ode_steps * F / length
                theta = theta + dt / ode_steps * vel

        return sequence, thetas, velocities

    sequences = []
    poss = []
    vels = []
    for i in range(train_set_size + valid_set_size + test_set_size):
        if i % 100 == 0:
            print(i)
        seq, ang, vel = generate_sequence()
        sequences.append(seq)
        poss.append(ang)
        vels.append(vel)
    sequences = np.array(sequences, dtype=np.uint8)
    poss = np.array(poss, dtype=np.float32)
    vels = np.array(vels, dtype=np.float32)

    np.savez_compressed(dest,
                        train_x={'frames': sequences[:train_set_size],
                                 'pos': poss[:train_set_size],
                                 'vel': vels[:train_set_size]},
                        valid_x={'frames': sequences[train_set_size:train_set_size+valid_set_size],
                                 'pos': poss[train_set_size:train_set_size+valid_set_size],
                                 'vel': vels[train_set_size:train_set_size+valid_set_size]},
                        test_x={'frames': sequences[train_set_size+valid_set_size:],
                                'pos': poss[train_set_size+valid_set_size:],
                                'vel': vels[train_set_size+valid_set_size:]})

    result = gallery(np.concatenate(sequences[:10] / 255), ncols=sequences.shape[1])

    norm = plt.Normalize(0.0, 1.0)
    fig, ax = plt.subplots(figsize=(sequences.shape[1], 10))
    ax.imshow(np.squeeze(result), interpolation='nearest', cmap=cm.Greys_r, norm=norm)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(dest.split(".")[0] + "_samples.jpg")


def generate_pendulum_intensity_dataset(dest,
                              train_set_size,
                              valid_set_size,
                              test_set_size,
                              seq_len,
                              img_size,
                              r=10,
                              length=10,
                              proj_dist=20,
                              max_theta=np.pi/4,
                              ode_steps=5,
                              dt=0.3,
                              mass=5):
    from skimage.draw import disk  # seems to be "disk" now, not "circle"
    from skimage.transform import resize

    def generate_sequence():
        sequence = []
        theta = np.random.uniform(-max_theta, max_theta)
        vel = 0
        thetas = []
        velocities = []
        for _ in range(seq_len):
            velocities.append(vel)
            thetas.append(theta)
            frame = np.zeros((img_size, img_size, 3))

            d = length * np.sin(theta)
            intensity = (proj_dist - length) ** 2 / (proj_dist - d) ** 2
            rr, cc = disk((img_size // 2, img_size // 2), r)
            frame[rr, cc, :] = (int(255 * intensity), 0, 0)
            frame = frame.astype(np.uint8)

            sequence.append(frame)

            for _ in range(ode_steps):
                F = -mass * 10 * np.sin(theta)
                vel = vel + dt / ode_steps * F / length
                theta = theta + dt / ode_steps * vel

        return sequence, thetas, velocities

    sequences = []
    poss = []
    vels = []
    for i in range(train_set_size + valid_set_size + test_set_size):
        if i % 100 == 0:
            print(i)
        seq, ang, vel = generate_sequence()
        sequences.append(seq)
        poss.append(ang)
        vels.append(vel)
    sequences = np.array(sequences, dtype=np.uint8)
    poss = np.array(poss, dtype=np.float32)
    vels = np.array(vels, dtype=np.float32)

    np.savez_compressed(dest,
                        train_x={'frames': sequences[:train_set_size],
                                 'pos': poss[:train_set_size],
                                 'vel': vels[:train_set_size]},
                        valid_x={'frames': sequences[train_set_size:train_set_size+valid_set_size],
                                 'pos': poss[train_set_size:train_set_size+valid_set_size],
                                 'vel': vels[train_set_size:train_set_size+valid_set_size]},
                        test_x={'frames': sequences[train_set_size+valid_set_size:],
                                'pos': poss[train_set_size+valid_set_size:],
                                'vel': vels[train_set_size+valid_set_size:]})

    result = gallery(np.concatenate(sequences[:10] / 255), ncols=sequences.shape[1])

    norm = plt.Normalize(0.0, 1.0)
    fig, ax = plt.subplots(figsize=(sequences.shape[1], 10))
    ax.imshow(np.squeeze(result), interpolation='nearest', cmap=cm.Greys_r, norm=norm)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(dest.split(".")[0] + "_samples.jpg")