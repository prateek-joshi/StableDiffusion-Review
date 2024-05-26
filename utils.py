import cv2
import numpy as np
import argparse
import glob
import os


def get_surface_normal_from_depth(depth, K=None):
    zy, zx = np.gradient(depth)
    normal = np.dstack((-zx, -zy, np.ones_like(depth)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n
    # offset and rescale values to be in 0-255
    normal += 1
    normal /= 2
    normal *= 255
    return normal

def get_canny_edges_from_depth(depth, kernel_size=5, alpha=30, beta=50):
    blurred = cv2.GaussianBlur(depth_map, (kernel_size, kernel_size), 0)
    edges = cv2.Canny(blurred, alpha, beta)
    return edges


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_source',
        required=True,
        type=str
    )
    parser.add_argument(
        '--dest',
        required=False,
        default='results'
    )
    parser.add_argument(
        '--canny',
        required=False,
        action='store_true',
        help='Run canny edge detection on source images.'
    )
    parser.add_argument(
        '--normal',
        required=False,
        action='store_true',
        help='Run surface normals detection on source images.'
    )
    args = parser.parse_args()
    assert os.path.isdir(args.dest)
    paths = []
    if os.path.isdir(args.img_source):
        paths = glob.glob(os.path.join(args.img_source, "*.{jpg,png,jpeg,npy}"))
    else:
        paths.append(args.img_source)
    for path in paths:
        if path.endswith((".jpg", ".png", ".jpeg")):
            depth_map = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        elif path.endswith(('.npy', '.npz')):
            depth_map = np.load(path).astype(np.uint8)
        extension = os.path.basename(path)
        if args.canny:
            res = get_canny_edges_from_depth(depth_map)
            save_name = os.path.basename(path)
            save_name = save_name.replace(f'.{save_name.split(".")[-1]}')
            save_path = os.path.join(args.dest_dir, '')