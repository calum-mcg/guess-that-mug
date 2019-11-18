# Guess That Mug: Live facial and gesture recognition
## Summary
Live facial and gesture recognition in Python using OpenCV, scikit-learn and TensorFlow.

This project was built with the aim of allowing users to become a human 'remote' - for example, playing/stopping personalised music playlists.

<p align="center">
  <img src="https://i.imgur.com/fuWiUNY.gif" width="650">
</p>

The project allows the user to build their own bespoke facial and hand gesture recognition models using both Scikit-learn and TensorFlow. This is achieved in three steps:
1. Capture facial and hand gesture images for training the models
2. Build the facial recognition model (SVM with Scikit) and the gesture recognition model (DNN with TensorFlow)
3. Run webcam feed and use models to predict faces and gestures on the fly

Prebuilt models from OpenCV are used to predict the presence of faces. Gesture recognition is limited to an area within a bounding box, whereas facial recognition is not.

Please read the __Installation & Usage__ section below on how to install and run the model. If interested, I have also detailed what goes on underneath the hood in the __How it works__ section. 

## Installation & Usage
Firstly install the required packages by running the following:

```
pip install -r requirements.txt
```

I have provided three example scripts to show how to use the classes.
1. [Capture facial and hand gesture images](../master/gather_images.py)
2. [Build (and save) both the facial recognition model and gesture recognition model](../master/process_images.py)
3. [Predict both faces and gestures live](../master/predict_live.py)

__*Please note:*__ When storing the images for both faces and gestures, the folders are used as labels to train the respective models. Please ensure the structure is the same as the following:
<p align="center">
  <img src="https://i.imgur.com/spJe275.png" width="850">
</p>

## How it works
### Process
The following process flows show how the class captures images. For gestures, background subtraction is used around the bounding box to identify the gesture. [OpenCV](https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html) has an insightful article with more details.
<p align="center">
  <img src="https://i.imgur.com/iGsprvJ.png" width="900">
</p>
For faces, the prebuilt OpenCV model is used to identify each face within the frame.
<p align="center">
  <img src="https://i.imgur.com/8MU1VKv.png" width="900">
</p>

The following process flows show how the class builds the models:
<p align="center">
  <img src="https://i.imgur.com/uGj0xE4.png" width="900">
</p>
<p align="center">
  <img src="https://i.imgur.com/maDc1Cc.png" width="900">
</p>

The following process flow shows how the class predicts faces and gestures live:
<p align="center">
  <img src="https://i.imgur.com/fBEOt5l.png" width="900">
</p>

## References
* [OpenCV](https://opencv.org/)
* [FaceNet Research](https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf)
* [Pyimagesearch](https://www.pyimagesearch.com/)
* Digital Image Processing - Rafael C. Gonzalez & Richard E. Woods

