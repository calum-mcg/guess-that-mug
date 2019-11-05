from model.ImageProcessing import *

gesture_images= './images/Gestures'
face_images = './images/Faces'

# Initialize weight for running average
a_weight = 0.65
# Specify threshold
threshold = 50

prebuilt_prototxt = './prebuilt_model/deploy.prototxt.txt'
prebuilt_model = './prebuilt_model/res10_300x300_ssd_iter_140000.caffemodel'
prebuilt_embedding_model = './prebuilt_model/openface_nn4.small2.v1.t7'

min_face_confidence = 0.75
face_name = "Example"
output_folder = './outputs/'

ImageModel = ImageProcessing(gesture_images, face_images, a_weight, threshold, prebuilt_prototxt, prebuilt_model,
                             min_face_confidence, face_name, prebuilt_embedding_model, output_folder)
ImageModel.process_faces()
ImageModel.process_gestures()