# CXR-CTR-UncertaintyEstimation

This project analyzes Chest X-Ray (CXR) images to calculate the Cardiothoracic (CT) ratio and its uncertainty.

## Installation and Setup

1. Clone this repository.

2. Build and run the Docker image:
   ```
   docker build -t cxr-ctrue .
   docker run --gpus '"device=0"' -it --ipc=host -v <local_data_path>:/workspace cxr_ctrue
   ```

3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### 1. DICOM to PNG Conversion

Convert CXR images from DICOM format to PNG:

```
python cxr_dcm2png.py --data_dir <dicom_input_dir> --savepoint <png_output_dir>
```

- `<dicom_input_dir>`: Path to the directory containing DICOM files
- `<png_output_dir>`: Path to the directory where PNG files will be saved

### 2. CT Ratio and Uncertainty Calculation

Calculate the CT ratio and uncertainty using the converted PNG images:

```
python CXRSegment_values.py --dataroot <png_input_dir> --mask_name_ls 'lung' 'heart' --savepoint './' --weights <model_weights_path> --resize_factor 512 --background --model 'Unet' --encoder 'tu-efficientnet_b4' --activation 'softmax2d'
```

- `<png_input_dir>`: Path to the directory containing converted PNG files
- `<model_weights_path>`: Path to the pre-trained model weights file

## Notes

- It's recommended to use absolute paths for all directory and file locations.
- Results will be saved in the specified `--savepoint` directory.

