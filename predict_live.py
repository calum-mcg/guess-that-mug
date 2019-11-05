from model.LivePrediction import *

# Initialize weight for running average when identifying background part of image
a_weight = 0.65
# Specify threshold for image subtraction with gestures
threshold = 50

# Prebuilt models and prototxt
prebuilt_prototxt = './prebuilt_model/deploy.prototxt.txt'
prebuilt_model = './prebuilt_model/res10_300x300_ssd_iter_140000.caffemodel'
prebuilt_face_model = './prebuilt_model/res10_300x300_ssd_iter_140000.caffemodel'
prebuilt_embedding_model = './prebuilt_model/openface_nn4.small2.v1.t7'

# Confidence levels for identifying faces / gesture
min_face_confidence = 0.75
min_gesture_confidence = 0.75

# Minimum frame count on face recognition before action taken
min_face_frame_count = 30

# Output folder where previously built models reside
output_folder = './outputs/'

# Create LivePrediction class
LiveModel = LivePrediction(output_folder, min_gesture_confidence, min_face_confidence, min_face_frame_count, a_weight,
                           threshold, prebuilt_prototxt, prebuilt_face_model, prebuilt_embedding_model)

# Run live
LiveModel.run_live()

