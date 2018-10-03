#
#
#
#
# OUT OF DATE !!!
# Please use run-preprocessed.py instead
#
#
#
#
import os
import tensorflow as tf
print "TensorFlow version: " + tf.__version__

dev=False

# Reduce logging verbosity to solve https://stackoverflow.com/questions/52512381/disable-image-parsing-warnings-in-tensorflow-python
#os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
#tf.logging.set_verbosity(tf.logging.ERROR)

thumbnail_height=20
thumbnail_width=20

out_shape = tf.convert_to_tensor([thumbnail_height, thumbnail_width])
batch_size = 100

if dev:
    data_folders = ["data/dev/training/0", "data/dev/training/1"]
else:
    data_folders = ["data/real/training/0", "data/real/training/1"]

classes = [0., 1.]

file_names = [] # Path of all data files
labels = [] # Label of each data file (same size as the array above)
for d, l in zip(data_folders, classes):
    name = [os.path.join(d,f) for f in os.listdir(d)] # get the list of all the images file names
    file_names.extend(name)
    labels.extend([l] * len(name))

epoch_size = 5
print "file_names: " + str(file_names)
print "labels: " + str(labels)
print "epoch_size: " + str(epoch_size)

file_names = tf.convert_to_tensor(file_names, dtype=tf.string)
labels = tf.convert_to_tensor(labels)

dataset = tf.data.Dataset.from_tensor_slices((file_names, labels))
dataset = dataset.repeat().shuffle(epoch_size)

def map_fn(path, label):
    # path/label represent values for a single example
    image = tf.image.decode_jpeg(tf.read_file(path), channels=3)

    # some mapping to constant size - be careful with distorting aspect ratios
    image = tf.image.resize_images(image, out_shape)
    image = tf.image.rgb_to_grayscale(image) # Improvement: Use Gleam grayscale, better for face recognition: https://stackoverflow.com/a/29989836/226958 https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0029740

    # color normalization - just an example
    image = tf.to_float(image) * (2. / 255) - 1
    label = tf.expand_dims(label, axis=-1)
    return image, label

# num_parallel_calls > 1 induces intra-batch shuffling
dataset = dataset.map(map_fn, num_parallel_calls=8)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)
print "dataset: " + str(dataset)

images, labels = dataset.make_one_shot_iterator().get_next()

# Following is from https://www.tensorflow.org/tutorials/keras/basic_classification
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(thumbnail_height, thumbnail_width, 1)),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print "images: " + str(images)
print "labels: " + str(labels)
model.fit(images, labels, epochs=epoch_size, verbose=1, steps_per_epoch=1095)

# Test

batch_size_test = 100

if dev:
    data_folders_test = ["data/dev/test/0", "data/dev/test/1"]
else:
    data_folders_test = ["data/real/test/0", "data/real/test/1"]

file_names_test = [] # Path of all data files
labels_test = [] # Label of each data file (same size as the array above)
for d_test, l_test in zip(data_folders_test, classes):
    name_test = [os.path.join(d_test,f_test) for f_test in os.listdir(d_test)] # get the list of all the images file names
    file_names_test.extend(name_test)
    labels_test.extend([l_test] * len(name_test))
epoch_size_test = 1
print "file_names: " + str(file_names_test)
print "labels: " + str(labels_test)
print "epoch_size: " + str(epoch_size_test)
dataset_test = tf.data.Dataset.from_tensor_slices((file_names_test, labels_test))
dataset_test = dataset_test.repeat().shuffle(epoch_size_test)
# num_parallel_calls > 1 induces intra-batch shuffling
dataset_test = dataset_test.map(map_fn, num_parallel_calls=8)
dataset_test = dataset_test.batch(len(file_names_test))
dataset_test = dataset_test.prefetch(1)
print "dataset: " + str(dataset_test)
images_test, labels_test = dataset_test.make_one_shot_iterator().get_next()
loss_test, accuracy_test = model.evaluate(images_test, labels_test, steps=1)
print('Test accuracy:', accuracy_test)

# Print prediction and certaincy for each image

predictions = model.predict(images_test, steps=1)

f = open("test-results.csv", "w")
for i, prediction in enumerate(predictions):
    name = file_names_test[i]
    label = str(file_names_test[i][15:16])
    prediction_str = str(prediction[0])
    f.write(prediction_str + "," + label + "," + name + "\n")
    #print "Prediction: " + str(prediction) + " Actual: " + str(tf.keras.backend.get_value(labels_test[i])[0]) + " File: " + file_names_test[i]
    print "Prediction: " + prediction_str + " Actual: " + label + " File: " + name
    #print labels_test[i][0]
    #with tf.Session() as sess:
    #  print labels_test[i].eval()
