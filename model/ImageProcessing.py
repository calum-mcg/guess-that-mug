from imutils.video import VideoStream
import numpy as np
import imutils
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import cv2
from sklearn.utils import shuffle
from imutils import paths
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

class ImageProcessing:
	# Class contains methods to take store images from webcam
	def __init__(self, gesture_output, face_output, a_weight, threshold, prebuilt_prototxt, prebuilt_model,
	             min_face_confidence, face_name, prebuilt_embedding_model, output_folder):
		# Display initial message
		print('[INFO] Constructing Image Processing class...')

		self.gesture_output = gesture_output
		self.face_output = face_output
		self.a_weight = a_weight
		self.threshold = threshold
		self.prebuilt_prototxt = prebuilt_prototxt
		self.prebuilt_model = prebuilt_model
		self.min_face_confidence = min_face_confidence
		self.face_name = face_name
		self.prebuilt_embedding_model = prebuilt_embedding_model

		self.face_embeddings = output_folder + 'face_embeddings.pickle'
		self.gesture_model_output = output_folder + 'gesture_model.tfl'
		self.face_model_output = output_folder + 'recogniser_model.pickle'
		self.face_label_output = output_folder + 'label_encoder.pickle'

	def capture_faces(self):
		# Store number of images saved
		total = 0

		# Load pre-built face detection model
		print("[INFO] Loading prebuilt face model...")
		net = cv2.dnn.readNetFromCaffe(self.prebuilt_prototxt, self.prebuilt_model)

		# Start video stream
		print("[INFO] Starting face capture video stream...")
		vs = VideoStream(src=0).start()

		# Inform user of keypresses required
		print("[INFO] Press 'q' to quit and 'k' to store an image...")

		while True:
			# Capture frame-by-frame
			frame = vs.read()
			# Copy frame for later storage if required
			orig = frame.copy()
			# Resize video stream
			frame = imutils.resize(frame, width=400)

			# Convert frame to a blob
			(h, w) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

			# Pass blob to model and get any face detections
			net.setInput(blob)
			detections = net.forward()

			# Draw a rectangle around the faces, by looping through identified faces and drawing box with label
			for i in range(0, detections.shape[2]):
				# Get prediction confidence to be used as a label
				confidence = detections[0, 0, i, 2]

				# Filter out any weak predictions using arbitrary constant
				if confidence < self.min_face_confidence:
					continue

				# Compute the (x, y)-coordinates of the bounding box
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# Draw box and label with accuracy
				label = "{:.2f}%".format(confidence * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
				cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

			# Display the resulting frame
			cv2.imshow('Frame', frame)

			key = cv2.waitKey(1) & 0xFF

			# If 'k' pressed store image, with 'name' in filename and add to total
			if key == ord("k"):
				p = os.path.sep.join([self.face_output, "{}-{}.png".format(self.face_name, str(total).zfill(5))])
				print(p)
				cv2.imwrite(p, orig)
				total += 1
			# If 'q' pressed, break loop
			if key == ord('q'):
				break

		# Print total number
		print("[INFO] {} face images stored".format(total))

		# Destroy windows and stop video stream
		cv2.destroyAllWindows()
		vs.stop()

	def capture_gestures(self):
		# Store number of images saved
		total = 0

		# Set background
		bg = None

		print("[INFO] Starting gesture capture video stream...")

		# Get the reference to the webcam
		camera = cv2.VideoCapture(0)

		# Inform user of keypresses required
		print("[INFO] Press 'q' to quit and 'k' to store an image...")

		# Region of interest (ROI) coordinates
		top, right, bottom, left = 100, 500, 300, 700

		# Initialise num of frames
		num_frames = 0

		while True:
			# Get the current frame
			(grabbed, frame) = camera.read()

			# Resize the frame
			frame = imutils.resize(frame, width=700)

			# Flip the frame so that it is not the mirror view
			frame = cv2.flip(frame, 1)

			# Clone the frame
			clone = frame.copy()

			# Get the height and width of the frame
			(height, width) = frame.shape[:2]

			# Get the ROI
			roi = frame[top:bottom, right:left]

			# Gonvert the roi to grayscale and blur it
			gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
			gray = cv2.GaussianBlur(gray, (7, 7), 0)

			# To get the background, keep looking until a threshold is reached
			# so that our running average model gets calibrated
			if num_frames < 30:
				if bg is None:
					bg = gray.copy().astype("float")
				else:
					# Compute weighted average, accumulate it and update the background
					cv2.accumulateWeighted(gray, bg, self.a_weight)
			else:
				# Segment the hand region
				# find the absolute difference between background and current frame
				diff = cv2.absdiff(bg.astype("uint8"), gray)

				# threshold the diff image so that we get the foreground
				thresholded = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)[1]

				# Get the contours in the thresholded image
				(cnts, _) = cv2.findContours(thresholded.copy(),
				                             cv2.RETR_EXTERNAL,
				                             cv2.CHAIN_APPROX_SIMPLE)

				# Return None, if no contours detected
				if len(cnts) == 0:
					hand = None
				else:
					# Based on contour area, get the maximum contour which is the hand
					segmented = max(cnts, key=cv2.contourArea)
					hand = (thresholded, segmented)
				# Check whether hand region is segmented
				if hand is not None:
					# If yes, unpack the thresholded image and
					# segmented region
					(thresholded, segmented) = hand

					# Draw the segmented region and display the frame
					cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
					cv2.imshow("Thesholded", thresholded)
					orig = thresholded.copy()

			# Draw the segmented hand
			cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

			# Increment the number of frames
			num_frames += 1

			# Display the frame with segmented hand
			cv2.imshow("Video Feed", clone)

			# Observe the keypress by the user
			keypress = cv2.waitKey(1) & 0xFF

			# If user press 'k' save image
			if keypress == ord("k"):
				p = os.path.sep.join([self.gesture_output, "{}.png".format(str(total).zfill(5))])
				print(p)
				cv2.imwrite(p, orig)
				total += 1
			# if the user press 'q' stop looping
			if keypress == ord("q"):
				break

		# Print total number
		print("[INFO] {} gesture images stored".format(total))

		# Destroy windows and stop video stream
		camera.release()
		cv2.destroyAllWindows()

	def process_faces(self):
		total = 0
		known_embeddings = []
		known_names = []

		# load our serialized face detector from disk
		print("[INFO] Loading face detector...")
		detector = cv2.dnn.readNetFromCaffe(self.prebuilt_prototxt, self.prebuilt_model)

		# load our serialized face embedding model from disk
		print("[INFO] Loading face recognizer...")
		embedder = cv2.dnn.readNetFromTorch(self.prebuilt_embedding_model)

		# grab the paths to the input images in our dataset
		print("[INFO] Quantifying faces...")
		image_paths = list(paths.list_images(self.face_output))

		# loop over the image paths
		for (i, imagePath) in enumerate(image_paths):
			# extract the person name from the image path
			print("[INFO] Processing face image {}/{}".format(i + 1, len(image_paths)))
			name = imagePath.split(os.path.sep)[-2]

			# load the image, resize it to have a width of 600 pixels (while
			# maintaining the aspect ratio), and then grab the image
			# dimensions
			image = cv2.imread(imagePath)
			image = imutils.resize(image, width=600)
			(h, w) = image.shape[:2]

			# construct a blob from the image
			image_blob = cv2.dnn.blobFromImage(
				cv2.resize(image, (300, 300)), 1.0, (300, 300),
				(104.0, 177.0, 123.0), swapRB=False, crop=False)

			# apply OpenCV's deep learning-based face detector to localize
			# faces in the input image
			detector.setInput(image_blob)
			detections = detector.forward()

			# ensure at least one face was found
			if len(detections) > 0:
				# we're making the assumption that each image has only ONE
				# face, so find the bounding box with the largest probability
				i = np.argmax(detections[0, 0, :, 2])
				confidence = detections[0, 0, i, 2]

				# ensure that the detection with the largest probability also
				# means our minimum probability test (thus helping filter out
				# weak detections)
				if confidence > self.min_face_confidence:
					# compute the (x, y)-coordinates of the bounding box for
					# the face
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					# extract the face ROI and grab the ROI dimensions
					face = image[startY:endY, startX:endX]
					(fH, fW) = face.shape[:2]

					# ensure the face width and height are sufficiently large
					if fW < 20 or fH < 20:
						continue

					# construct a blob for the face ROI, then pass the blob
					# through our face embedding model to obtain the 128-d
					# quantification of the face
					face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					                                  (96, 96), (0, 0, 0), swapRB=True, crop=False)
					embedder.setInput(face_blob)
					vec = embedder.forward()

					# add the name of the person + corresponding face
					# embedding to their respective lists
					known_names.append(name)
					known_embeddings.append(vec.flatten())
					total += 1

		# Dump the facial embeddings + names to disk
		print("[INFO] Serializing {} face encodings...".format(total))
		data = {"embeddings": known_embeddings, "names": known_names}
		f = open(self.face_embeddings, "wb")
		f.write(pickle.dumps(data))
		f.close()

		# encode the labels
		print("[INFO] Encoding face labels...")
		le = LabelEncoder()
		labels = le.fit_transform(data["names"])

		# train the model used to accept the 128-d embeddings of the face and
		# then produce the actual face recognition
		print("[INFO] Training face model...")
		recognizer = SVC(C=1.0, kernel="linear", probability=True)
		recognizer.fit(data["embeddings"], labels)

		print("[INFO] Saving face model...")
		# write the actual face recognition model to disk
		f = open(self.face_model_output, "wb")
		f.write(pickle.dumps(recognizer))
		f.close()

		# write the label encoder to disk
		f = open(self.face_label_output, "wb")
		f.write(pickle.dumps(le))
		f.close()

	@staticmethod
	def resize_image(image_path):
		base_width = 100
		image = Image.open(image_path)
		wpercent = (base_width / float(image.size[0]))
		hsize = int((float(image.size[1]) * float(wpercent)))
		img = image.resize((base_width, hsize), Image.ANTIALIAS)
		img.save(image_path)

	def process_gestures(self):
		total = 0

		image_paths = list(paths.list_images(self.gesture_output))

		loaded_images = []
		gesture_labels = []

		# Loop over images, resizing and converting to numpy array
		# Also storing labels
		for (i, image_path) in enumerate(image_paths):
			self.resize_image(image_path)
			gesture_name = image_path.split(os.path.sep)[-2]
			if gesture_name == "OK":
				gesture_label = [1, 0]
			elif gesture_name == "Palm":
				gesture_label = [0, 1]
			else:
				raise ValueError('Gesture name unknown.')
			gesture_labels.append(gesture_label)
			gesture_image = cv2.imread(image_path)
			gray_image = cv2.cvtColor(gesture_image, cv2.COLOR_BGR2GRAY)
			loaded_images.append(gray_image.reshape(100, 100, 1))
			print("[INFO] Processing gesture {}. ({} - {})".format(i, gesture_name, image_path))

		# Define the CNN Model
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

		# Shuffle data in unison
		loaded_images, output_vectors = shuffle(loaded_images, gesture_labels, random_state=0)

		# Split into test and training data
		X_train, X_test, y_train, y_test = train_test_split(loaded_images, output_vectors, test_size=0.2,
		                                                    random_state=0)
		print("[INFO] Training gesture model...")
		# Train model
		model.fit(X_train, y_train, n_epoch=50, validation_set=(X_test, y_test),
		          snapshot_step=100, show_metric=True, run_id='convnet_coursera')
		print("[INFO] Saving gesture model...")
		model.save(self.gesture_model_output)
