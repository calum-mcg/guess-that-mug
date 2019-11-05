from model.ImageProcessing import ImageProcessing
resize_image = ImageProcessing.resize_image
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from imutils.video import FPS
import numpy as np
import imutils
import pickle
import cv2


class LivePrediction:
	def __init__(self, output_folder, min_gesture_confidence, min_face_confidence, min_face_frame_count, a_weight,
	             threshold, prebuilt_prototxt, prebuilt_face_model, prebuilt_embedding_model):
		# Display initial message
		print('[INFO] Constructing Live Prediction class...')

		self.min_gesture_confidence = min_gesture_confidence
		self.min_face_confidence = min_face_confidence
		self.min_face_frame_count = min_face_frame_count
		self.a_weight = a_weight
		self.threshold = threshold

		self.prebuilt_prototxt = prebuilt_prototxt
		self.prebuilt_face_model = prebuilt_face_model
		self.prebuilt_embedding_model = prebuilt_embedding_model

		self.face_embeddings = output_folder + 'face_embeddings.pickle'
		self.gesture_model_output = output_folder + 'gesture_model.tfl'
		self.face_model_output = output_folder + 'recogniser_model.pickle'
		self.face_label_output = output_folder + 'label_encoder.pickle'

	def run_live(self):
		# Set background
		background = None
		# Region of interest (ROI) coordinates
		top, right, bottom, left = 100, 500, 300, 700
		# Initialise num of frames
		num_frames = 0

		# Load face detector
		print("[INFO] Loading prebuilt face detector...")
		detector = cv2.dnn.readNetFromCaffe(self.prebuilt_prototxt, self.prebuilt_face_model)
		# Load face embedding model
		print("[INFO] Loading prebuilt face recognizer...")
		embedder = cv2.dnn.readNetFromTorch(self.prebuilt_embedding_model)
		# Load built face recognition model along with the label encoder
		print("[INFO] Loading built model and label encoder from pickle...")
		recognizer = pickle.loads(open(self.face_model_output, "rb").read())
		le = pickle.loads(open(self.face_label_output, "rb").read())
		# Get number of users
		number_users = len(le.classes_)
		# Create empty array for counting frames recognised of users
		user_frame_count = np.array([[0] * self.min_face_frame_count for i in range(number_users)])

		# Empty label text & gesture name
		gesture_label = ""
		gesture_name = ""
		face_label = ""
		face_name = ""
		full_label = ""

		# Define the CNN Model for gesture recognition using TensorFlow
		tf.reset_default_graph()
		convnet = input_data(shape=[None, 100, 100, 1], name='input')
		convnet = conv_2d(convnet, 32, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)
		convnet = conv_2d(convnet, 64, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)
		convnet = conv_2d(convnet, 128, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)
		convnet = conv_2d(convnet, 256, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)
		convnet = conv_2d(convnet, 256, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)
		convnet = conv_2d(convnet, 128, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)
		convnet = conv_2d(convnet, 64, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)
		convnet = fully_connected(convnet, 1000, activation='relu')
		convnet = dropout(convnet, 0.75)
		convnet = fully_connected(convnet, 2, activation='softmax')
		convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
		                     name='regression')
		model = tflearn.DNN(convnet, tensorboard_verbose=0)

		# Load saved gesture model
		print("[INFO] Loading gesture model...")
		model.load(self.gesture_model_output)

		# Get VideoCapture
		camera = cv2.VideoCapture(0)

		# Start the FPS estimator
		fps = FPS().start()

		# Inform user of keypresses required
		print("[INFO] Press 'q' to quit...")

		# Loop until keypress
		while True:
			# Get the current frame
			(grabbed, frame) = camera.read()

			# Resize the frame to have width 700 pixels
			frame = imutils.resize(frame, width=700)

			# Get the height and width of the frame
			(h, w) = frame.shape[:2]

			# Construct a blob from the image
			image_blob = cv2.dnn.blobFromImage(
				cv2.resize(frame, (300, 300)), 1.0, (300, 300),
				(104.0, 177.0, 123.0), swapRB=False, crop=False)

			# Firstly identify face then identify gesture
			# Apply OpenCV's deep learning-based face detector to localize
			# faces in the input image
			detector.setInput(image_blob)
			detections = detector.forward()
			# Get count of recognised faces
			detections_count = np.array([[0] for i in range(number_users)])
			# Loop over facial detections
			for i in range(0, detections.shape[2]):
				# Extract the confidence associated with face prediction
				face_confidence = detections[0, 0, i, 2]
				# filter out weak detections
				if face_confidence > self.min_face_confidence:
					# Compute the coordinates of the bounding box for the face
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					# Extract the face ROI
					face = frame[startY:endY, startX:endX]
					(fH, fW) = face.shape[:2]

					# Ensure the face width and height are sufficiently large
					if fW < 20 or fH < 20:
						continue

					# Construct a blob for the face ROI, then pass the blob
					# through the face embedding model to obtain the 128-d
					# quantification of the face
					face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					                                  (96, 96), (0, 0, 0), swapRB=True, crop=False)
					embedder.setInput(face_blob)
					vec = embedder.forward()

					# Perform classification to recognize the face
					preds = recognizer.predict_proba(vec)[0]
					j = np.argmax(preds)
					proba = preds[j]
					face_name = le.classes_[j]
					# Increase count of specific face recognised for min_frame recognise requirement
					detections_count[j] = [1]
					# Draw the bounding box of the face along with the associated probability
					face_label = "{} ({:.2f}%)".format(face_name, proba * 100)
					y = startY - 10 if startY - 10 > 10 else startY + 10
					# Display green box if face recognised, not "Unknown" class and frame count reached
					if (sum(user_frame_count[j]) == self.min_face_frame_count) and (face_name != "Unknown"):
						cv2.rectangle(frame, (endX, startY), (startX, endY), (50, 205, 50), 2)
					else:
						cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
				# Once all faces recognised remove first column and add observations to last
				user_frame_count = np.delete(user_frame_count, 0, 1)
				user_frame_count = np.append(user_frame_count, detections_count, axis=1)
			# Secondly identify gestures
			# Flip the frame so it's not the mirror view (otherwise confusing when recognising gestures)
			frame = cv2.flip(frame, 1)
			# Clone the frame
			clone = frame.copy()
			# Get the ROI
			roi = frame[top:bottom, right:left]

			# Convert the roi to grayscale and blur it
			gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
			gray = cv2.GaussianBlur(gray, (7, 7), 0)

			# to get the background, keep looking till a threshold is reached
			# so that our running average model gets calibrated
			if num_frames < 30:
				if background is None:
					background = gray.copy().astype("float")
				else:
					# Compute weighted average, accumulate it and update the background using weighting
					cv2.accumulateWeighted(gray, background, self.a_weight)
			else:
				# Segment the hand region
				# Find the absolute difference between background and current frame
				diff = cv2.absdiff(background.astype("uint8"), gray)

				# Threshold the diff image so that we get the foreground
				thresholded = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)[1]

				# Get the contours in the thresholded image
				(cnts, _) = cv2.findContours(thresholded.copy(),
				                             cv2.RETR_EXTERNAL,
				                             cv2.CHAIN_APPROX_SIMPLE)

				# Return None if no contours detected
				if len(cnts) == 0:
					hand = None
				else:
					# Based on contour area, get the maximum contour which is the hand
					segmented = max(cnts, key=cv2.contourArea)
					hand = (thresholded, segmented)
				# Check whether hand region is segmented
				if hand is not None:
					# Unpack the thresholded image and segmented region
					(thresholded, segmented) = hand

					# Draw the segmented region and display the frame
					cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
					cv2.imshow("Thesholded", thresholded)
					cv2.imwrite('Temp.png', thresholded)
					resize_image('Temp.png')
					image = cv2.imread('Temp.png')
					gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
					prediction = model.predict([gray_image.reshape(100, 100, 1)])
					prediction_index = np.argmax(prediction)
					prediction_confidence = np.amax(prediction) / (prediction[0][0] + prediction[0][1])
					if prediction_confidence > self.min_gesture_confidence:
						if prediction_index == 0:
							gesture_name = "OK"
						elif prediction_index == 1:
							gesture_name = "Palm"
						gesture_label = "{} ({:.2f}%)".format(gesture_name, prediction_confidence * 100)
					else:
						gesture_name = "Unknown"
						gesture_label = "{}".format(gesture_name)

			# Draw the segmented hand
			cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

			# Display predictions in bottom bar
			# Green text if both face and gesture known
			full_label = face_label + " " + gesture_label
			if (gesture_name == "Unknown") or (face_name == "Unknown"):
				cv2.putText(clone, full_label, (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
			else:
				cv2.putText(clone, full_label, (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 205, 50), 2)

			# Increment the number of frames
			num_frames += 1

			# Update the FPS counter
			fps.update()

			# Display the segmented hand in separate window
			cv2.imshow("Video Feed", clone)

			# Store keypress
			keypress = cv2.waitKey(1) & 0xFF

			# If "q" pressed then break loop
			if keypress == ord("q"):
				break

		# Stop the timer and display FPS information
		fps.stop()
		print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
		print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

		# Free up memory
		camera.release()
		cv2.destroyAllWindows()
