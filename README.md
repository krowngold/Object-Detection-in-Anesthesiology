<!-- # CSE 455 Final Project - Machine Learning Models for Syringe Identification -->
## Lauren He and Noah Krohngold

The main goals for this project were:
1. Optimizing existing YOLO models used by the University of Washington to identify syringes in frames
2. Automating image annotation for further training
3. Detecting text labels on syringes and extracting the text for verification of medicines used

### Problem Description
Manual entry of medications administered intra-operatively is often inaccurate and potentially detracts from patient monitoring, particularly during periods of induction and emergence. Additionally, most medications are recorded after delivery, making it hard to intervene if an inappropriate drug is selected. One potential solution involves real-time detection and recording of drug administration using smart eyewear and computer vision. For our project, we will train a neural network to identify syringes and vials in an anesthesia provider's hand and drug draw up events using a head-mounted camera to work towards establishing this first of its kind advanced warning system. 

### Previous Work
IRB approval was obtained for intra-operative recordings using a head-mounted camera worn by anesthesiology providers. Images were collated and labeled by a group of trained annotators. The original dataset was based on 28 clips from 5 days of recordings, sampling 6 frames per second for a total of 3200 images. Data included images captured under regular-light and low-light conditions. The images were tiled to 1/8th the original size and only tiles that contained at least 75% of the syringe were included in training and validation. 

The YOLOv5 algorithm was used given its speed and real-time object analysis capabilities. Training and validation were performed on both low-light and regular-light tiled images for 200 epochs per case (as no further improvement was observed beyond a certain point). Training was initially done with default YOLOv5 settings (image size 640 and pre-trained weights) on full-size and tiled images. Three additional training scenarios were trialed on full-size and tiled images: 1. image size 640 without pre-trained weights, 2. image size 416 with pretrained weights, 3. image size 416 without pre-trained weights. Testing of the neural network was completed on tiled low- and regular-light images and full-sized low-light images.

The model performed worse under low light conditions. A question that may be asked is: why not increase the light in these conditions. Certain surgeries are performed in low light for various reasons. Low light settings could make it easier to see contrast in tissues or contrast agents such as fluorescent dyes which mark cancer cells. Laparoscopic surgeries rely on video screens since surgery is done through small holes rather than a big incision, and the view is from a camera inserted into one of the small holes. Increasing the light in these rooms would make it harder for surgeons to accurately diagnose tissue such as identifying malignant tumors from benign tumors or even perform the operation itself. There are some small lights at the anesthesia workstation, but a headlamp or excess light source would pose a distraction. 
