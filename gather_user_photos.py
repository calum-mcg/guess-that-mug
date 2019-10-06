from imutils.video import VideoStream
import numpy as np
import imutils
import os
import cv2


# Set parameters required
prototxt = './prebuilt_model/deploy.prototxt.txt'
model = './prebuilt_model/res10_300x300_ssd_iter_140000.caffemodel'
min_confidence = 0.6
output = ".\image"
total = 0
name = "Calum2"

# Load pre-built face detection model
print("Loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Start video stream
print("Starting video stream...")
vs = VideoStream(src=0).start()

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
		if confidence < min_confidence:
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
		p = os.path.sep.join([output, "{}-{}.png".format(name, str(total).zfill(5))])
		print(p)
		cv2.imwrite(p, orig)
		total += 1
	# If 'q' pressed, break loop
	if key == ord('q'):
		break

# Print total number
print(" {} face images stored".format(total))

# Destroy windows and stop video stream
cv2.destroyAllWindows()
vs.stop()
