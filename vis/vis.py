import numpy as np
import torch
import matplotlib.pyplot as plt


##################################################
directory_path = "./vis/"
bev_size = 200
roi = (-51.2, 51.2)


##################################################
def load_data():
    save = torch.load(directory_path + "save.pt")

    current_gt = save['current_gt']
    previous_preds = save['previous_preds']

    return current_gt, previous_preds

def visualize_previous_preds(previous_preds):
    norm_coords = (previous_preds[:, :2] - roi[0]) / (roi[1] - roi[0])
    bev_coords = (norm_coords * bev_size).to(torch.long)

    spots = torch.zeros(size=(200, 200), dtype=torch.long)
    spots[bev_coords[:, 1], bev_coords[:, 0]] = 1
    spots = spots.numpy()

    plt.imshow(spots, cmap='gray', interpolation='none')

def draw_a_bbox(bbox):
    x, y, _, w, l, _, rot, _, _ = bbox

    x_ = (x - roi[0]) / (roi[1] - roi[0])
    y_ = (y - roi[0]) / (roi[1] - roi[0])
    w_ = w / (roi[1] - roi[0])
    l_ = l / (roi[1] - roi[0])

    x_ = x_ * bev_size
    y_ = y_ * bev_size
    w_ = w_ * bev_size
    l_ = l_ * bev_size

    init_corners = np.array([[-w_/2, -l_/2], [-w_/2, l_/2], [w_/2, l_/2], [w_/2, -l_/2]])
    rotation_matrix = np.array([[np.cos(-rot), -np.sin(-rot)], [np.sin(-rot), np.cos(-rot)]])
    exact_corners = np.array([x_, y_]) + (init_corners @ rotation_matrix.T)

    plt.plot(*np.vstack((exact_corners, exact_corners[0])).T, color='r', linewidth=1)


##################################################
current_gt, previous_preds = load_data()

plt.figure(figsize=(10, 10))
plt.xlim(0, bev_size)
plt.ylim(0, bev_size)

for bbox in current_gt:
    draw_a_bbox(bbox)

visualize_previous_preds(previous_preds)

plt.savefig(directory_path + "result.png", dpi=200, bbox_inches='tight', pad_inches=0)
plt.close()