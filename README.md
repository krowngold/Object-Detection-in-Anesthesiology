<!-- # CSE 455 Final Project - Machine Learning Models for Syringe Identification -->
## Lauren He and Noah Krohngold

The main goals for this project were:
1. Optimizing existing object detection model to identify syringes in real time
2. Automating image annotation for further training
3. Detecting text labels on syringes and extracting the text for verification of medicines used

### Problem Description
IRB approval was obtained for intra-operative recordings using a head-mounted camera worn by anesthesiology providers. Images were collated and labeled by a group of trained annotators. The original dataset was based on 28 clips from 5 days of recordings, sampling 6 frames per second for a total of 3200 images. Data included images captured under regular-light and low-light conditions. The images were tiled to 1/8th the original size and only tiles that contained at least 75% of the syringe were included in training and validation. 

The YOLOv5 framework was used given its speed and accessibility. Training and validation were performed on both low-light and regular-light tiled images for 200 epochs per case (as no further improvement was observed beyond a certain point). Training was initially done with default YOLOv5 settings (image size 640 and pre-trained weights) on full-size and tiled images. Three additional training scenarios were trialed on full-size and tiled images: 1. image size 640 without pre-trained weights, 2. image size 416 with pretrained weights, 3. image size 416 without pre-trained weights. Testing of the neural network was completed on tiled low- and regular-light images and full-sized low-light images.

The model performed worse under low light conditions. A question that may be asked is: why not increase the light in these conditions. Certain surgeries are performed in low light for various reasons. Low light settings could make it easier to see contrast in tissues or contrast agents such as fluorescent dyes which mark cancer cells. Laparoscopic surgeries rely on video screens since surgery is done through small holes rather than a big incision, and the view is from a camera inserted into one of the small holes. Increasing the light in these rooms would make it harder for surgeons to accurately diagnose tissue such as identifying malignant tumors from benign tumors or even perform the operation itself. There are some small lights at the anesthesia workstation, but a headlamp or excess light source would pose a distraction.


### Approach
The prior iteration of the project used a pretrained model with out of the box parameters (YOLOv5x). We planned on experimenting with different settings to try to improve upon the current model. The original images from our dataset were resized while maintaining the aspect ratio where the longer side is set to the width/length of the resized square and padding is added to the shorter side. The previous project used 416 and 640 image sizes. For our model, we only used an image size of 640. We increased the batch size from 16 to 32. We are now identifying 3 classes (Vial, DrugDrawUp, Syringe) instead of 1 (Syringe). Our larger dataset comprises 4419 images instead of 3200. 

For tuning our model, we planned to use orthogonalization which is a process of using controls that are aligned with one aspect of performance without affecting another. If the training set does not fit well which suggests some degree of bias, we intend to test out a larger network (YOLOv5x6), use more data, or add additional layers to the model. We might use an adaptive learning rate optimizer where there’s one learning rate for each parameter instead of SGD. If the validation set does not fit well, we would use a larger validation set or increase the batch size to 64 or 128 for added regularization. Lastly, we will test our detector on clips it has not seen before. 

### Datasets
We currently have a collection of around 70 annotated clips that each have about 1000 frames with 300-400 images. For this project, we pulled a subset of those clips. From those clips, we used a full-size set of frames and a tiled set. The clips are grouped based on lighting conditions: well lit, dim, dark, and very dark. 
<style>
.tablelines table, .tablelines td, .tablelines th {
        border: 1px solid black;
        }
</style>
| 1 (very dark) | 2 (dark)   | 3 (dim)    | 4 (well lit) |
| ------------- | ---------  | ---------- | ------------ |
| train: 306    | train: 802 | train: 323 | train: 1075  |
| valid: 90     | valid: 78  | valid: -   | valid: 593   |
| test: 209     | test: 542  | test: -    | test: 402    |  
{: .tablelines}  

![Annotated data](/example_images.png)  
Figure 1: Top left: Syringe, Top right: DrugDrawUp, Bottom left: Vial, Bottom right: DrugDrawUp

The full-size set has a total of 4420 images with 2506 in train, 761 in val, and 1153 in test. The MATLAB program that was used to match each frame of annotation with the images decides the folder (train/valid/test) that each clip gets put into randomly at runtime. But since lighting level 3 was such a small dataset, it didn't generate a valid and test folder. We usually put 70% of the clips in train, 20% in val, and 10% in test. For this project, we used an 80-10-10 split. Our initial run incorrectly split on images. In the following run, we split on clips and combined train and test sets from each light level to form the final training set (3659 images) while the validation set was set aside for validation (761 images). 

### Results

The results from our initial model were invalid because our data was split on images so frames from the same clips were spread across train, val, and test. The model converged very quickly and reached an mAP@.5 of more than 90% near epoch 20. And from there, the mAP continued to climb. The best model had an mAP@.5 of 0.98033 and an mAP@.5:.95 of 0.798.

![Train/Val losses and metrics](/losses_and_metrics.png)  

Figure 2: Result plots for the first run split on images

With our best model, we ran inference on two clips the model has not seen before (one well lit and one low lit) to demonstrate model functionality. The model performed rather well on the well lit clip and performed noticeably worse on the low lit clip. For the well lit clip, it was able to predict high confidence boxes of the right classes rather consistently. We thresholded objectness confidence at 40%. When the detector did predict boxes in the low lit clip, it predicted them with high confidence but at a lower rate. There were likely predictions that did not meet the threshold and therefore was not shown in the output making it appear that the detector is not predicting often enough. We do see that fewer boxes appear in the low lit inference clip than the well lit inference clip even though the objects of interest are within the frame. We think that this model performed relatively well because the clips inherently have low variance in that they were captured in specific environments at the university and the identification task itself is narrow. 

The final model was trained on data split by clips and was trained for 599 epochs. We originally planned to train for 1000 epochs. We did not train further due to time limitations and the model seemed to be oscillating around a local minimum. The best model had an mAP@.5 of 0.787 and an mAP@.5:.95 of 0.569. Similar to the first model, this model converged rapidly and plateaued. As expected, it doesn’t reach as high of an accuracy reflecting the correction in the split. In both cases, recall lags behind precision.

![Result plots for run split on clips](/figure3data.png)  

Figure 3: Result plots for the run split on clips

### Discussion
- What problems were encountered?

We encountered several problems. Our initial approach was to tune a select number of hyperparameters one and a time. Because we would have had to first handpick some hyperparameters and then tune them independently, we were interested in utilizing the hyperparameter evolution feature that comes with YOLOv5 which is a method of hyperparameter optimization that uses a genetic algorithm to find the optimal values of hyperparameters. After some investigation, we decided not to move forward with hyperparameter evolution deeming it too expensive and time consuming. It was not the lifeline we thought it would be as it is recommended to run the base scenario 300 times. We would have had to run tens of thousands if not hundreds of thousands of epochs.  

We originally planned to develop capabilities for real time identification during low-lit conditions by performing image preprocessing to increase brightness in low-lit images prior to identifying objects. However, we learned that it will likely not work as we intend because of batch normalization which is a critical component of neural nets. Batch norm transforms activations over a mini-batch by taking the mean of the mini-batch and subtracting the mean from each unit and dividing by the standard deviation. This step would actually increase the brightness of images without us having to perform it beforehand. In a way, it would be undoing our preprocessing. 

To our human eyes, the dim image and the same image with increased brightness look appreciably different but we are not increasing the amount of information available to the computer. In fact, preprocessing the image might introduce degradation like the glare from a light spot on a dim image we were testing that became too bright after processing. We did, however, include low lit images when training as the network should learn on its own how to deal with such images. 

Setting up training represented a significant time sink. We came across some operational issues using the Azure Machine Learning Studio interface, particularly dealing with latency around uploading and shuttling around files and folders. Our dataset was compressed into rar files and required unpacking. The rar files were actually corrupted on the first download which was not picked up during unpacking in the Azure notebook. Unpacking locally revealed the underlying checksum errors. We had to re-download the rar files, unpack them locally, and then upload them to Azure.

Once we had our data prepped for training, we immediately came up against a memory bottleneck and had to decrease batch size. This thwarted our plan to speed up training with a larger batch size. In general, limited time and resources were the major blockers in completing this project. We were fortunate to have access to multi-core GPUs through a scholarship. Even with that, it was difficult to do extensive experimentation.


- Are there next steps you would take if you kept working on the project?

Because of time and resource constraints, we were not able to perform the number of experiments needed for rapid iteration. In order to do quick prototyping, we could use a smaller dataset. We suspect that the uniformity of our data would lead the model to be unable to generalize well. To increase the diversity of our dataset, we could adjust and add to the hyperparameters that control data augmentation which is baked into the YOLOv5 package.
```
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # image HSV-Value augmentation (fraction)
degrees: 0.0  # image rotation (+/- deg)
translate: 0.1  # image translation (+/- fraction)
scale: 0.5  # image scale (+/- gain)
shear: 0.0  # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.0  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mosaic: 1.0  # image mosaic (probability)
mixup: 0.0  # image mixup (probability)
```
Figure 4: YOLOv5 data augmentation hyperparameters

In future iterations, we will experiment with the tiled dataset which we were unable to run a model against within the timeframe of this project. We will include no label during the event clips in our dataset so that the model can learn non-objects of interest. We could adjust the complexity of the model by using a smaller pre-trained checkpoint as our final model does seem to be overfitting. Since YOLOv5 doesn’t introduce new architectural changes, we plan on diving into the model architecture of YOLOv4 as it did improve upon the performance of YOLOv3. One stretch goal we were not able to investigate was optical character recognition to identify drug labels and their contents as the ultimate goal of this work is to figure out what drug is in the syringe.

We could vary the objectness confidence threshold to fit different use cases. We likely care more about higher object confidence in well lit settings and could use lower object confidence to get more bounding boxes in low lit settings. We will continue to investigate the possibility of using semi-supervised learning to automate annotations using our final detector.

- How does your approach differ from others? Was that beneficial?

The differences in the approach we used compared to the prior project are outlined in the approach section. We ran fewer experiments given we were time and resource-constrained and used full-size images and a larger dataset. We were not able to achieve a higher accuracy measure than the best model from the prior project. However, our dataset combined well lit and low lit images to train on whereas the earlier project trained separate models on well lit and low lit images. From those results, we know that detection on low lit images is harder which could explain why our model performs worse than the model trained on tiled well lit images, pre-trained weights, and an image size of 640 (mAP@.5 of 0.865). There is as of yet no conclusive evidence that training on tiled images in general yields better performance than training on full-size images. In the previous experiment, the models trained on tiled low lit images did perform significantly better than the models trained on full-size low lit images. For the model trained on pre-trained weights and an image size of 640, the mAP@.5 was 0.61 and 0.178, respectively.

