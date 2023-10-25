semantic restoration 
---

### content
1. [Environment]
3. [Download]
4. [How2predict]
5. [How2train]

## Environment
pytorch==1.2.0    

## Download
This paper has trained examples of transformations using the horse2zebra dataset as an example, and the trained generator and discriminator models are as follows:  
[Generator_A2B_horse2zebra.pth] 
[Generator_B2A_horse2zebra.pth]  
[Discriminator_A_horse2zebra.pth]
[Discriminator_B_horse2zebra.pth] 

Commonly used datasets are addressed below:   
link (on a website): https://pan.baidu.com/s/1xng_uQjyG-8CFMktEXRdEg extraction code: grtm     

## How2predict
### a、Using pre-trained weights
1. After downloading the library, unzip it and download the corresponding weights file and store it in model_data.
2. Run the predict.py file.
3. Input the path of the image to be predicted and get the prediction result.
### b、Use your own trained weights
1. Follow the training steps.    
2. Inside the model.py file, change the model_path to correspond to the trained file in the following section; **model_path corresponds to the weights file under the logs folder**    
```python
_defaults = {
    #-----------------------------------------------#
    #  model_path points to the weights file in the logs folder
    #-----------------------------------------------#
    "model_path"        : 'model_data/Generator_A2B_horse2zebra.pth',
    #-----------------------------------------------#
    #   Input image size setting
    #-----------------------------------------------#
    "input_shape"       : [128, 128],
    #-------------------------------#
    #   Whether to resize without distortion
    #-------------------------------#
    "letterbox_image"   : True,
    #-------------------------------#
    #   Whether to use Cuda
    #   No GPU can be set to False
    #-------------------------------#
    "cuda"              : True,
}
```
3. Run the predict.py file.
4. Input the path of the image to be predicted and get the prediction result.

## How2train
1. The image files that are expected to be converted are placed in the datasets folder before training, there are two categories, and the purpose of the training is to allow category A and B to be converted to each other.
2. Run txt_annotation.py under the root directory to generate train_lines.txt, making sure that there is a file path content inside train_lines.txt.  
3. Run the train.py file for training, the images generated during training can be viewed in the results/train_out folder.  
