from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
import pandas as pd
import numpy as np

TRAIN_DIR = 'data/train'  # Corrected path
TEST_DIR = 'data/test'  # Corrected path

def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir,label)):
            image_paths.append(os.path.join(dir,label,imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels

train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)
print(train)

test = pd.DataFrame()
test['image'], test['label'] = createdataframe(TEST_DIR)

def extract_features(images):
    features = []
    for image in images:
        img = load_img(image, color_mode="grayscale")
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features),48,48,1)
    return features

train_features = extract_features(train['image'])
test_features = extract_features(test['image'])

x_train = train_features/255.0
x_test = test_features/255.0

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train['label'])

y_train = le.transform(train['label'])
y_test = le.transform(test['label'])
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

model = Sequential()
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Corrected metrics parameter

model.fit(x=x_train,y=y_train, batch_size=128, epochs=10, validation_data=(x_test,y_test))

model_json = model.to_json()
with open("emotiondetector.json",'w') as json_file:
    json_file.write(model_json)
    model.save("emotiondetector.h5")

from keras.models import model_from_json
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

label = ['angry','disgust','fear','happy','neutral','sad','surprise']











# def ef(image):
#     img = load_img(image, color_mode="grayscale")
#     feature = np.array(img)
#     feature = feature.reshape(1,48,48,1)
#     return feature/255.0

# image = 'images/train/sad/im42.png'
# print("original image is of sad")
# img = ef(image)
# pred = model.predict(img)
# pred_label = label[pred.argmax()]
# print("model prediction is ",pred_label)

# image = 'images/train/disgusted/im299.png'
# print("original image is of disgust")
# img = ef(image)
# pred = model.predict(img)
# pred_label = label[pred.argmax()]
# print("model prediction is ",pred_label)

# image = 'images/test/happy/im27.png'
# print("original image is of happy")
# img = ef(image)
# pred = model.predict(img)
# pred_label = label[pred.argmax()]
# print("model prediction is ",pred_label)

# from keras.models import load_model
# from sklearn.metrics import accuracy_score, classification_report


# model = load_model('emotiondetector.h5')

# loss, accuracy = model.evaluate(x_test, y_test)
# print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

# y_pred = model.predict(x_test)
# y_pred_classes = np.argmax(y_pred, axis=1)
# y_true_classes = np.argmax(y_test, axis=1)
# y_pred_labels = le.inverse_transform(y_pred_classes)
# y_true_labels = le.inverse_transform(y_true_classes)

# label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# print("\nClassification Report:")
# print(classification_report(y_true_labels, y_pred_labels, target_names=label))
