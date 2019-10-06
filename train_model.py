from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

# Set parameters required
embeddings = './outputs/embeddings.pickle'
output_model = './outputs/recogniser_model.pickle'
output_label_encoder = './outputs/label_encoder.pickle'

# load the face embeddings
print("Loading face embeddings...")
data = pickle.loads(open(embeddings, "rb").read())

# encode the labels
print("Encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("Training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open(output_model, "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(output_label_encoder, "wb")
f.write(pickle.dumps(le))
f.close()
