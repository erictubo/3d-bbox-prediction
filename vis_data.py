import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# each subfolder in data contains the following:
# - bbox3d.npy: ground truth 3D bounding box
# - mask.npy: instance segmentation
# - pc.npy: point cloud
# - rgb.jpg: RGB image

# QUESTIONS
# - how is the image related to the point cloud?
    # - segmentation mask probably in image space
    # - bounding box is probably in 3D space
    # - any projection parameters available - or is this the core of the problem?


def visualize_image(image, mask, n=0):

    # show image only
    plt.imshow(image)
    plt.title("Image only")
    plt.show()

    N = mask.shape[0]
    assert n <= N

    plt.imshow(image)

    if n < N:
        plt.title(f"Image with {n}th mask (inverted)")
        # mask_inv = 1 - mask[n]
        masked_data = np.ma.masked_where(~mask[n], mask[n])
        plt.imshow(masked_data, alpha=0.8, cmap="Reds")

    if n == N:
        plt.title(f"Image with all {N} masks")
        
        cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
        alphas = [1.0] *6 + [0.8] *6 + [0.6] *6 + [0.4] *6
        c = len(cmaps)
        
        for i in range(N):
            masked_data = np.ma.masked_where(~mask[i], mask[i])
            plt.imshow(masked_data, alpha=alphas[i], cmap=cmaps[i % c], vmin=0, vmax=1)

        # background of all masks in light grey
        plt.imshow(np.ma.masked_where(np.sum(mask, axis=0) != 0, np.sum(mask, axis=0)), alpha=0.5, cmap="Greys", vmin=0, vmax=1)

    plt.show()


def visualize_point_cloud(point_cloud, bbox, n=0, limits=True):

    N = bbox.shape[0]
    assert n <= N

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # include origin in the plot
    ax.scatter(0, 0, 0, s=10, c='black', alpha=0.0)

    # set limits to the point cloud
    if limits:
        ax.set_xlim(np.min(point_cloud[0]), np.max(point_cloud[0]))
        ax.set_ylim(np.min(point_cloud[1]), np.max(point_cloud[1]))
        ax.set_zlim(np.min(point_cloud[2]), np.max(point_cloud[2]))

    # set initial view as camera perspective
    ax.view_init(elev=90, azim=90)

    if n < N: # show nth bbox
        ax.set_title(f'3D point cloud with {n}th bbox')
        ax.scatter(bbox[n, :, 0], bbox[n, :, 1], bbox[n, :, 2], s=10, c='red', alpha=1)

    if n == N: # show all bboxes (inverted and in different colors)
        ax.set_title(f'3D point cloud with all {N} bboxes')

        colors = ['grey', 'purple', 'blue', 'green', 'orange', 'red']
        alphas = [1.0] *6 + [0.8] *6 + [0.6] *6 + [0.4] *6
        
        for i in range(N):
            ax.scatter(bbox[i, :, 0], bbox[i, :, 1], bbox[i, :, 2], s=10, c=colors[i % len(colors)], alpha=alphas[i])

    plt.show()



def main():

    subfolder_names = np.random.choice(os.listdir("data"), 5, replace=False)
    # print(subfolder_names)

    for subfolder_name in subfolder_names:
        subfolder_path = Path("data") / subfolder_name

        image = plt.imread(subfolder_path / "rgb.jpg") # (H, W, 3) - RGB image
        mask = np.load(subfolder_path / "mask.npy") # (N, H, W) - N objects

        point_cloud = np.load(subfolder_path / "pc.npy") # (3, H, W) - 3D coordinates for each pixel
        bbox = np.load(subfolder_path / "bbox3d.npy") # (N, 8, 3) - N objects, 8 corners, 3D coordinates

        H, W = image.shape[:2]
        N = mask.shape[0] # number of objects
        print("Number of objects:", N, "H:", H, "W:", W, "AR:", round(H/W, 2))

        # number = np.random.randint(0, N)
        number = N

        assert mask.shape == (N, H, W)
        assert bbox.shape == (N, 8, 3)
        assert point_cloud.shape == (3, H, W)

        visualize_image(image, mask, number)

        visualize_point_cloud(point_cloud, bbox, number)


if __name__ == "__main__":
    main()