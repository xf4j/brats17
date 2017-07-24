from __future__ import division
import os, sys, glob
import numpy as np
import nibabel as nib
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
from skimage.transform import resize
from multiprocessing import Pool, cpu_count

def n4_correction(im_input):
    n4 = N4BiasFieldCorrection()
    n4.inputs.dimension = 3
    n4.inputs.input_image = im_input
    n4.inputs.bspline_fitting_distance = 300
    n4.inputs.shrink_factor = 3
    n4.inputs.n_iterations = [50, 50, 30, 20]
    n4.inputs.output_image = im_input.replace('.nii.gz', '_corrected.nii.gz')
    n4.run()

def batch_works(k):
    if k == n_processes - 1:
        paths = all_paths[k * int(len(all_paths) / n_processes) : ]
    else:
        paths = all_paths[k * int(len(all_paths) / n_processes) : (k + 1) * int(len(all_paths) / n_processes)]
        
    for path in paths:
        n4_correction(glob.glob(os.path.join(path, '*_t1.nii.gz'))[0])
        n4_correction(glob.glob(os.path.join(path, '*_t1ce.nii.gz'))[0])
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("Need at least the input data directory")
    input_path = sys.argv[1]
        
    all_paths = []
    for dirpath, dirnames, files in os.walk(input_path):
        if os.path.basename(dirpath)[0:7] == 'Brats17':
            all_paths.append(dirpath)
            
    n_processes = cpu_count()
    pool = Pool(processes=n_processes)
    pool.map(batch_works, range(n_processes))