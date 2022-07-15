# kvasir-sessile
In this work swin-net transformer is used for sementic segmentation of Polyp Dataset for Computer Aided Gastrointestinal Disease Detection.
The data set can be downloaded from the link given below 
```
https://datasets.simula.no/kvasir-seg/
```
The rest of the repo. is divided as follows
1. Requirements 
2. Traning and Inference 
3. Results 
## Requirements
The main library requirements for this projects are as follows 
* **Tensorflow**
* **Matplotlib**
* **Opencv**

The complete requirements can be installed using the following set of commands 

```
cd kvasir-sessile/
conda create  -n <environment -name> python==3.7.4
conda activate <environment-name>
pip install -r requirements.txt
```

## Traning and Inference

The traning and inference code is given inside the **Swin_UNET_128.ipynb**


## Results

The visual results of this work are shown in the following figure where first column shows the  ground truth sample images. The second coulmn shows their respective predictions . The third coulmn shows the error. between the predicted and the ground truth. 

![Alt text](./Results.png?raw=true "Title")
