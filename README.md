# kvasir-sessile
In this work we have used swin-net transformer for sementic segmentation of Polyp Dataset for Computer Aided Gastrointestinal Disease Detection.
The data set can be downloaded from the link given below 
```
https://datasets.simula.no/kvasir-seg/
```

## Traning and Inference
Before traning the model or performing inference create a conda  environment with the **python=3.7**  and then inside the environment install the  ** requirement.txt** using the the following command 

```

pip install -r requirement.txt
```
The traning and inference code is given inside the **Swin_UNET_128.ipynb**


## Results

The visual results of this work are shown in the following figure where first column shows the  ground truth sample images. The second coulmn shows their respective predictions . The third coulmn shows the error. between the predicted and the ground truth. 

![Alt text](./Results.png?raw=true "Title")