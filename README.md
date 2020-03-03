DC-EDN: Densely Connected Encoder-Decoders Network With Reinforced Depthwise Convolution for Face Alignment
-------------- 
DC-EDN structure  
--------------------
The DC-EDN(4) structure is showed: 
![](https://github.com/iam-zhanghongliang/DC-EDN/blob/master/picture/structure.png)

 
Environment configuration
---------------------
We use python 3.6.2 and Pytorch 1.01 during training and testing.


Validation
--------------------------
We currently only publicly release the code for the validation experiments on WFLW. Implementation details and results of validation are shown below.

Implementation details
------------------------
The first step is to download the WFLW dataset from that extract password is say2 URL:https://pan.baidu.com/s/1J9OsmxZR0LHl242O2NwWtg or  https://wywu.github.io/projects/LAB/WFLW.html 

The dataset includes 10,000 images of which 7,500 are training images and 2500 are test images.

The second step:

    (1).Adding the path of the dataset images downloaded in the second step(1) to img_folder = ("") of validate.py file 

    (2).Adding the path of test set in validate_dataset to LoadPicture("") and FACE("") of the validate.py file.

    (3).Adding the path of model(.tar) trained in model to torch.load("") of the validate.py file.

The third step is to run the code: python validate.py --exp_id cu-net-0  --bs 1

By running the code according to the steps, you will get the FR, AUC, CED curve and NME under different test sets.

results
-----------------
The results of compared with different excellent methods are shown (see the table below).
-----------------------------------------------------------------------------------------
![](https://github.com/iam-zhanghongliang/DC-EDN/blob/master/picture/result.png)

Visualizing results  
----------------------------
  !['The_new_block'](https://github.com/iam-zhanghongliang/DC-EDN/blob/master/picture/occlu1.png)!['The_new_block'](https://github.com/iam-zhanghongliang/DC-EDN/blob/master/picture/makeup1.png)!['The_new_block'](https://github.com/iam-zhanghongliang/DC-EDN/blob/master/picture/makeup2.png)
  
Future work
 ----------------
(1). The training code will be released on 300W and WFLW.

(2). The testing code will be released on 300W.
  
 Questions
 -------------
 Please contact iamzhanghongliang@163.com
