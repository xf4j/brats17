from __future__ import division
import os, sys, glob
import numpy as np
import nibabel as nib
from skimage import measure
from scipy.ndimage.morphology import binary_dilation
from multiprocessing import Pool, cpu_count

def clean_contour(prob, c_input):
    # Smaller areas with lower prob are very likely to be false positives
    wt_mor = binary_dilation((c_input > 0).astype(np.float32), iterations=10)
    labels = measure.label(wt_mor)
    w_area = []
    for l in range(1, np.amax(labels) + 1):
        w_area.append(np.sum(prob[labels == l]))
    if len(w_area) > 0:
        max_area = np.amax(w_area)
        for l in range(len(w_area)):
            if w_area[l] < max_area / 2.0:
                c_input[labels == l + 1] = 0
    return c_input

def batch_works(k):
    if k == n_processes - 1:
        paths = all_paths[k * int(len(all_paths) / n_processes) : ]
    else:
        paths = all_paths[k * int(len(all_paths) / n_processes) : (k + 1) * int(len(all_paths) / n_processes)]
        
    for path in paths:
        probs = np.load(os.path.join(input_path, path))
        pred = np.argmax(probs, axis=3)
        fg_prob = 1 - probs[..., 0]
        pred = clean_contour(fg_prob, pred)
        seg = np.zeros(pred.shape, dtype=np.int16)
        seg[pred == 1] = 1
        seg[pred == 2] = 2
        seg[pred == 3] = 4
        img = nib.Nifti1Image(seg, np.eye(4))
        nib.save(img, os.path.join(output_path, path.replace('_probs.npy', '.nii.gz')))
    
if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise Exception("Need at least the input and out data directory")
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    all_paths = os.listdir(input_path)
            
    n_processes = cpu_count()
    pool = Pool(processes=n_processes)
    pool.map(batch_works, range(n_processes))