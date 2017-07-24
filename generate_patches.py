from __future__ import division
import os, sys, glob
import numpy as np
from multiprocessing import Pool, cpu_count
from utils import generate_patch_locations, perturb_patch_locations, generate_patch_probs, read_image

def batch_works(k):
    if k == n_processes - 1:
        paths = all_paths[k * int(len(all_paths) / n_processes) : ]
    else:
        paths = all_paths[k * int(len(all_paths) / n_processes) : (k + 1) * int(len(all_paths) / n_processes)]
        
    for path in paths:
        o_path = os.path.join(output_path, os.path.basename(path))
        if not os.path.exists(o_path):
            os.makedirs(o_path)
        x, y, z = perturb_patch_locations(base_locs, patch_size / 16)
        probs = generate_patch_probs(path, (x, y, z), patch_size, image_size)
        selections = np.random.choice(range(len(probs)), size=patches_per_image, replace=False, p=probs)
        image = read_image(path)
        for num, sel in enumerate(selections):
            i, j, k = np.unravel_index(sel, (len(x), len(y), len(z)))
            patch = image[int(x[i] - patch_size / 2) : int(x[i] + patch_size / 2),
                          int(y[j] - patch_size / 2) : int(y[j] + patch_size / 2),
                          int(z[k] - patch_size / 2) : int(z[k] + patch_size / 2), :]
            f = os.path.join(o_path, str(num))
            np.save(f, patch)
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("Need at least the input data directory")
    input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = './patches'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Some variables
    patches_per_image = 400
    patch_size = 64
    image_size = (240, 240, 155)
    
    all_paths = []
    for dirpath, dirnames, files in os.walk(input_path):
        if os.path.basename(dirpath)[0:7] == 'Brats17':
            all_paths.append(dirpath)
    
    base_locs = generate_patch_locations(patches_per_image, patch_size, image_size)
    
    n_processes = cpu_count()
    pool = Pool(processes=n_processes)
    pool.map(batch_works, range(n_processes))