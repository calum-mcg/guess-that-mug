from model.ImageProcessing import *

# Folders for gesture and face images
gesture_images = './images/Gestures'
face_images = './images/Faces'

# Initialize weight for running average when identifying background part of image
a_weight = 0.65
# Specify threshold for image subtraction with gestures
threshold = 50

# Prebuilt models and prototxt
prebuilt_prototxt = './prebuilt_model/deploy.prototxt.txt'
prebuilt_model = './prebuilt_model/res10_300x300_ssd_iter_140000.caffemodel'
prebuilt_embedding_model = './prebuilt_model/openface_nn4.small2.v1.t7'

# Minimum face confidence before storing
min_face_confidence = 0.75

# Label of face
face_name = "Example"

# Folder where any model outputs are stored
output_folder = './outputs/'

# Create Image Model class
ImageModel = ImageProcessing(gesture_images, face_images, a_weight, threshold, prebuilt_prototxt, prebuilt_model,
                             min_face_confidence, face_name, prebuilt_embedding_model, output_folder)

# Capture and store face images
ImageModel.capture_faces()

# Capture and store gesture images
ImageModel.capture_gestures()


