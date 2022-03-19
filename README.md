<!-- # CSE 455 Final Project - Machine Learning Models for Syringe Identification -->
## Lauren He and Noah Krohngold

The main goals for this project were:
1. Optimizing existing object detection model to identify syringes in real time
2. Automating image annotation for further training
3. Detecting text labels on syringes and extracting the text for verification of medicines used

### Problem Description
Manual entry of medications administered intra-operatively is often inaccurate and potentially detracts from patient monitoring, particularly during periods of induction and emergence. Additionally, most medications are recorded after delivery, making it hard to intervene if an inappropriate drug is selected. One potential solution involves real-time detection and recording of drug administration using smart eyewear and computer vision. For our project, we will train a neural network to identify syringes and vials in an anesthesia provider's hand and drug draw up events using a head-mounted camera to work towards establishing this first of its kind advanced warning system. 

### Previous Work
IRB approval was obtained for intra-operative recordings using a head-mounted camera worn by anesthesiology providers. Images were collated and labeled by a group of trained annotators. The original dataset was based on 28 clips from 5 days of recordings, sampling 6 frames per second for a total of 3200 images. Data included images captured under regular-light and low-light conditions. The images were tiled to 1/8th the original size and only tiles that contained at least 75% of the syringe were included in training and validation. 

The YOLOv5 algorithm was used given its speed and real-time object analysis capabilities. Training and validation were performed on both low-light and regular-light tiled images for 200 epochs per case (as no further improvement was observed beyond a certain point). Training was initially done with default YOLOv5 settings (image size 640 and pre-trained weights) on full-size and tiled images. Three additional training scenarios were trialed on full-size and tiled images: 1. image size 640 without pre-trained weights, 2. image size 416 with pretrained weights, 3. image size 416 without pre-trained weights. Testing of the neural network was completed on tiled low- and regular-light images and full-sized low-light images.

The model performed worse under low light conditions. A question that may be asked is: why not increase the light in these conditions. Certain surgeries are performed in low light for various reasons. Low light settings could make it easier to see contrast in tissues or contrast agents such as fluorescent dyes which mark cancer cells. Laparoscopic surgeries rely on video screens since surgery is done through small holes rather than a big incision, and the view is from a camera inserted into one of the small holes. Increasing the light in these rooms would make it harder for surgeons to accurately diagnose tissue such as identifying malignant tumors from benign tumors or even perform the operation itself. There are some small lights at the anesthesia workstation, but a headlamp or excess light source would pose a distraction. 


### Approach
The prior iteration of the project used a pretrained model with out of the box parameters (YOLOv5x). We planned on experimenting with different settings to try to improve upon the current model. The original images from our dataset were resized while maintaining the aspect ratio where the longer side is set to the width/length of the resized square and padding is added to the shorter side. The size was previously set to 416. For our model, we use an image size of 640. We increased the batch size from 16 to 32. We are now identifying 3 classes (Vial, DrugDrawUp, Syringe) instead of 1 (Syringe). Our larger dataset comprises 4419 images instead of 3200. 

For tuning our model, we planned to use orthogonalization which is a process of using controls that are aligned with one aspect of performance without affecting another. If the training set does not fit well which suggests some degree of bias, we intend to test out a larger network (YOLOv5x6), use more data, or add additional layers to the model. We might use an adaptive learning rate optimizer where there’s one learning rate for each parameter instead of SGD. If the validation set does not fit well, we would use a larger validation set or increase the batch size to 64 or 128 for added regularization. Lastly, we will test our detector on clips it has not seen before. 

### Datasets
We have a set of full-size images and tiled images. The clips are grouped based on lighting conditions: well lit, dim, dark, and very dark.
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



For full-size images, the set based on light conditions includes 2506 training images, 761 validation images, and 1153 test images. This set has a total of 4420 images. We usually put 70% of the clips in train, 20% in valid and 10% in test. For this project, we used an 80-10-10 split. The matlab program that was used to match each frame of annotation with the images decides the folder (train/valid/test) that each clip gets put into randomly at runtime. But since lighting level 3 was such a small dataset, it didn't generate a valid and test folder. Train and test sets from each light level were combined to form the final train set (3659 images) while the validation set was set aside for validation (761 images). 


### Results

![Train/Val losses and metrics](/losses_and_metrics.png)


### Discussion
- What problems were encountered?

We encountered several problems. Our initial approach was to tune a select number of hyperparameters one and a time. We were interested in using a feature of YOLOv5 called hyperparameter evolution which is a method of hyperparameter optimization that uses a genetic algorithm to find the optimal values of hyperparameters. After some investigation, we decided not to move forward with the plan to use hyperparameter evolution deeming it too expensive and time consuming. It was not the lifeline we thought it would be. It is recommended to run the base scenario 300 times. We would have had to run tens of thousands if not hundreds of thousands of epochs.  

We originally planned to develop capabilities for real time syringe identification during low-lit conditions by performing image preprocessing to increase brightness in low light images prior to identifying syringes. However, we learned that it will likely not work as we intend because of batch normalization which is a critical component of neural nets. Batch norm transforms activations over a mini-batch by taking the mean of the mini-batch and subtracting the mean from each unit and dividing by the standard deviation. This step would actually increase the brightness of images without us having to perform it beforehand. In a way, it would be undoing our preprocessing. 

To our human eyes, the dim image and the same image with increased brightness look significantly different but we are not increasing the amount of information available to the computer. In fact, preprocessing the image might introduce degradation like the glare from a light spot on a dim image we were testing that got too bright after processing. We did, however, include low light images when training as the network should learn on its own how to deal with such images. 

Uploading and shuttling around images and files and folders headaches 
Rar files, unpacking, checksum errors, re-downloading

Batch size limitation - impossibly long training time
Memory bottleneck - need to decrease batch size

- Are there next steps you would take if you kept working on the project?

We currently have around 70 annotated clips that each have about 1000 frames with 300-400 annotated images. For the final project, we used a subset of those images. 

Because of time and resource constraints, we were not able to perform the number of experiments needed for rapid iteration. In order to do quick prototyping, we would use a smaller dataset.

Random cropping or horizontal flipping - move bounding boxes with image (do this in data loader)
Random perturbations: exposure, color, saturation
Data augmentation is baked into the YOLOv5 package.
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

In future iterations, we should include no label during the event (true negative examples) in our dataset so that the model can learn what is not an object of interest.

Smaller model (if the model seems to be overfitting), ensembling

Change threshold to be greater than 0.5

Automate annotations using final detector

One stretch goal we were not able to attend to was exploring optical character recognition to identify drug labels as the ultimate goal of the work is to figure out what drug is in the syringe.

We investigated if it would be possible to leverage frame by frame information to automatically place bounding boxes using semi-supervised learning. 

- How does your approach differ from others? Was that beneficial?

The differences in the approach we used compared to the prior project are outlined above. We ran fewer experiments given we were time and resource-constrained using full-size images and a larger dataset. We were not able to achieve a higher accuracy measure than the current best model.

