# Guess That Mug: Live facial and gesture recognition
## Summary
Multiple classes built in Python that predict faces and hand gestures, live through a webcam feed. This project was built with the aim of allowing users to become a human 'remote' - for example, playing/stopping personalised music playlists.

<p align="center">
  <img src="https://i.imgur.com/fuWiUNY.gif" width="650">
</p>

The project has been built in Python and allows the user to build their own facial and hand gesture recognition models using both Scikit-learn and TensorFlow. This is achieved in three steps:
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
1. [Capture facial and hand gesture images](capture_images.py)
2. [Build (and save) both the facial recognition model and gesture recognition model](process_images.py)
3. [Predict both faces and gestures live](predict_live.py)

__*Please note:*__ When storing the images for both faces and gestures, the folders are used as labels to train the respective models. Please ensure the structure is the same as the following:
<p align="center">
  <img src="https://i.imgur.com/spJe275.png" width="850">
</p>

## How it works
### Process


## References
* [OpenCV](https://opencv.org/)
* [FaceNet Research](https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf)
* [Pyimagesearch](https://www.pyimagesearch.com/)
* Digital Image Processing - Rafael C. Gonzalez & Richard E. Woods

