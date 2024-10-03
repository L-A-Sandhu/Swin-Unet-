# Swin-Unet: Semantic Segmentation for Polyp Detection Using Transformer Models

This repository implements **Swin-Unet**, a powerful transformer-based architecture used for **semantic segmentation** of the **Kvasir Polyp Dataset**. The project aims to support **Computer-Aided Gastrointestinal Disease Detection** by accurately identifying and segmenting polyps in endoscopic images.

## Dataset
You can download the **Kvasir-SEG dataset** used in this project from the following link:  
[Kvasir-SEG Polyp Dataset](https://datasets.simula.no/kvasir-seg/)

The rest of the repo. is divided as follows
1. Requirements
2.  Data Prepration
3. Traning and Inference 
4. Results 

## Requirements

To run the **Swin-Unet** model for polyp detection and segmentation, you need to install the following main libraries:

- **TensorFlow**: For building and training the Swin-Unet transformer model.
- **Matplotlib**: For visualizing results and plotting data.
- **OpenCV**: For image processing tasks, including reading and preparing the dataset.

### Installation Instructions:

Follow these steps to set up the environment and install all dependencies:

1. **Clone the repository**:
   ```bash
cd Swin-unet/
conda create  -n <environment -name> python==3.7.4
conda activate <environment-name>
pip install -r requirements.txt
```
## Data Preparation

Before training the **Swin-Unet** model for **polyp segmentation**, you need to prepare the dataset.

### Steps to Prepare the Data:

1. **Download the Kvasir-SEG Dataset**:  
   Download the dataset from the following link:  
   [Kvasir-SEG Dataset](https://datasets.simula.no/kvasir-seg/)

2. **Extract the Data**:  
   After downloading, extract the dataset to your local machine.

3. **Organize the Files**:
   - Create a new folder called `data` inside the `Swin-unet` project directory.
   - Copy the images and masks from the **Kvasir-SEG** dataset into the `data` folder.

4. **Run Preprocessing**:  
   Use the following command to preprocess the dataset and convert the masks into PNG format:
   ```bash
   python preprocess.py

## Training and Inferene 
This work genrates a swin-unet model from scratch. However, it can save checkpoint, resume traning , test on data and perform infrence on a dataset.
Please follow the following set of commands 

### Train
```
python Swin_UNET_128.py  --data=<data set path>--class=< number of classes>  --inps= < Train, test, Resume>  --b_s=< batch size> --e=< Epoch> --p= < patience>  --model_dir= < Check_Point>
Example Command
python Swin_UNET_128.py  --data=./data --class=2 --inps=train --b_s=16 --e=10 --p=10 --model_dir='./checkpoint/'



```
### Test
```
python Swin_UNET_128.py  --data=<data> --class=< number of classes>  --inps= < train, test, resume> --model_dir='./checkpoint/'
Example Command
python Swin_UNET_128.py  --data=./data --class=2 --inps=test --model_dir='./checkpoint/'

```
### Resume 
```
python Swin_UNET_128.py  --data=<data set path>--class=< number of classes>  --inps= < Train, test, Resume>  --b_s=< batch size> --e=< Epoch> --p= < patience>  --model_dir= < Check_Point>

Example Command

python Swin_UNET_128.py  --data=./data --class=2 --inps=resume --b_s=16 --e=10 --p=10 --model_dir='./checkpoint/'

```
### Infer
```
python Swin_UNET_128.py  --data=< data for infrence > --class=2 --inps=< test, train, resume, infer> --model_dir=< checkp point>

python Swin_UNET_128.py  --data=./data --class=2 --inps=infer --model_dir='./checkpoint/'

```



## Results
The prediction accuracy, latency , Flops, size on disk and the number  of parameter are shown in the following table 
| Model         | Parameters | Accuracy | Latency(sec)   | Size on Disk (MB)| Flops |
|---------------|-------------|----------|----------------|------------------|-------|
| Swin-unet     | 3,783,510   | 0.836    |  0.0004        |        38.89     |0.681 G|


The visual results of this work are shown in the following figure where first column shows the  ground truth sample images. The second coulmn shows their respective predictions . The third coulmn shows the error. between the predicted and the ground truth. 
![Alt text](./Results.png?raw=true "Title")

