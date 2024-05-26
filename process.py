import cv2
import tqdm
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
    canny_savedir = os.path.join(args.dest, 'canny')
    normal_savedir = os.path.join(args.dest, 'normal')
    # depth_savedir = os.path.join(args.dest, 'depth')
    valid_extensions = (".jpg", ".png", ".jpeg", ".npy")
    paths = []
    if os.path.isdir(args.img_source):
        for ext in valid_extensions:
          # print(os.path.join(args.img_source, f"*{ext}"))
          paths += glob.glob(os.path.join(args.img_source, f"*{ext}"))
    else:
        paths.append(args.img_source)
    for path in tqdm.tqdm(paths, total=len(paths)):
        if path.endswith((".jpg", ".png", ".jpeg")):
            depth_map = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        elif path.endswith(('.npy', '.npz')):
            depth_map = np.load(path) * 2
            depth_map = depth_map.astype(np.uint8)
        # os.makedirs(depth_savedir, exist_ok=True)
        save_name = os.path.basename(path).split('.')[0] + '.png'
        # save_path = os.path.join(depth_savedir, save_name)
        # cv2.imwrite(save_path, depth_map)
        extension = os.path.splitext(path)[-1]
        if args.canny:
            os.makedirs(canny_savedir, exist_ok=True)
            res = get_canny_edges_from_depth(depth_map)
            # save_name = os.path.basename(path).split('.')[0] + '.png'
            # save_name = save_name.replace(extension, '-canny'+extension)
            save_path = os.path.join(canny_savedir, save_name)
            # print(save_path)
            cv2.imwrite(save_path, res)
        if args.normal:
            os.makedirs(normal_savedir, exist_ok=True)
            res = get_surface_normal_from_depth(depth_map)
            # save_name = os.path.basename(path).split('.')[0] + '.png'
            # save_name = save_name.replace(extension, '-normal'+extension)
            save_path = os.path.join(normal_savedir, save_name)
            # print(save_path)
            cv2.imwrite(save_path, res)