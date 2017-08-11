## Synopsis

Submission for Multimodal Brain Tumor Segmentation Challenge 2017  (http://braintumorsegmentation.org/). A patch-based 3D U-Net model is used. Instead of predicting the class label of the center pixel, this model predicts the class label for the entire patch. A sliding-window method is used in deployment with overlaps between patches to average the predictions.

## Code Example

The workflow includes bias correction, patch extraction, training, post-processing, testing and submission.</br></br>
After training data is downloaded, run `python bias_correction.py input_dir` to perform bias field correction based on N4ITK (https://www.ncbi.nlm.nih.gov/pubmed/20378467). The corrected dataset will be saved at the same folder with the raw dataset.</br></br>
Run `python generate_patches.py input_dir output_dir` to generate patches for training.</br></br>
To train the model, run `python main.py --train=True --train_data_dir=train_patch_dir`. Or you can modify the default parameters in `main.py` so that you can just run `python main.py`. Check `model.py` for more details about the network structure.<br/></br>
To test the model on validation dataset, run `python main.py --train=False --deploy_data_dir=deploy_data_dir --deploy_output_dir=deploy_output_dir`. The results will be saved at `deploy_output_dir`. The network structure for survival prediction is not working good as the result is similar as random guessing. So you can ignore that by setting `run_survival` to `False`.<br/></br>
To combine the results and generate the final label maps, run `python prepare_for_submission.py input_dir output_dir`.

## Installation

The model is implemented and tested using `python 2.7` and `Tensorflow 1.1.0`, but `python 3` and newer versions of `Tensorflow` should also work.
Other required libraries include: `numpy`, `h5py`, `skimage`, `transforms3d`, `nibabel`, `scipy`, `nipype`. You also need to install `ants` for bias correction. Read the instructions for Nipype (http://nipy.org/nipype/0.9.2/interfaces/generated/nipype.interfaces.ants.segmentation.html) and Ants (http://stnava.github.io/ANTs/) for more information.

## Contributors

Xue Feng, Department of Biomedical Engineering, University of Virginia
xf4j@virginia.edu