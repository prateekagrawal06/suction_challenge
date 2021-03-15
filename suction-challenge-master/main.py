import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Cropping2D, ZeroPadding2D, Activation, Add
from tqdm import tqdm
import glob
import os
import shutil
import numpy as np
import cv2
import argparse


def tf_read_image(img_obj, image_h, image_w, channels, norm=True):
    """
    Function to use tensorflow for reading the images
    :param img_obj: Path to the image to be decoded
    :param image_h: Height to resize the image to
    :param image_w: Width to resize the image to
    :param channels: Number to channel to decode the image into
    :param norm: Whether to normalize the image by dividing with 255.0
    :return: Returns a resized image with floating point values
    """
    x_img_string = tf.io.read_file(img_obj)
    x_img = tf.image.decode_png(x_img_string, channels=channels)
    x_img = tf.image.resize(x_img, (image_h, image_w))
    if norm:
        x_img = tf.math.divide(x_img, 255.0)
    return x_img


def read_images(color_path, depth_path, label_path, image_h, image_w):
    """
    Fucntion to read images for color, depth and labels
    :param color_path: Path to the color image
    :param depth_path: Path to the depth image
    :param label_path: Path to the label image
    :param image_h: Height to resize the images to
    :param image_w: Width to resize the images to
    :return: returns pixels values for each of the images
    """
    color = tf_read_image(color_path, image_h, image_w, channels=0, norm=True)
    depth = tf_read_image(depth_path, image_h, image_w, channels=0, norm=True)
    label = tf_read_image(label_path, image_h, image_w, channels=0, norm=True)
    return color, depth, label


def create_data(train_data, val_data):
    """
    Function to parse the text file with information on data
    :param train_data: Path to the text file with training images
    :param val_data: Path to the text file with validation images
    :return: python list storing the path to color, depth and label for training and validation
    """
    train_data_color_path = []
    train_data_depth_path = []
    trian_data_label_path = []

    val_data_color_path = []
    val_data_depth_path = []
    val_data_label_path = []

    train_file = open(train_data, 'r')
    lines = train_file.readlines()
    for line in lines:
        c, d, l, _ = line.split("png")
        train_data_color_path.append("../" + "/".join(c.split("/")[11:]) + "png")
        train_data_depth_path.append("../" + "/".join(d.split("/")[11:]) + "png")
        trian_data_label_path.append("../" + "/".join(l.split("/")[11:]) + "png")

    val_file = open(val_data, 'r')
    lines = val_file.readlines()
    for line in lines:
        c, d, l, _ = line.split("png")
        val_data_color_path.append("../" + "/".join(c.split("/")[11:]) + "png")
        val_data_depth_path.append("../" + "/".join(d.split("/")[11:]) + "png")
        val_data_label_path.append("../" + "/".join(l.split("/")[11:]) + "png")
    return train_data_color_path, train_data_depth_path, trian_data_label_path, val_data_color_path, \
           val_data_depth_path, val_data_label_path


def log_loss(loss, acc, precision, epoch, writer):
    """
    Fuction to log the loss to tensorboard
    :param loss: The loss tensor
    :param acc: the accuracy tensor
    :param precision: the precision tensor
    :param epoch: the epoch currently running
    :param writer: the summary writer object
    :return: None
    """
    with writer.as_default():
        tf.summary.scalar('loss', loss.result().numpy(), epoch)
        tf.summary.scalar('Accuracy', acc.result().numpy(), epoch)
        tf.summary.scalar('Precision', precision.result().numpy(), epoch)


def save_best_weights(model, name, loss):
    """
    Fucntion to save the model with the best weights
    :param model: Model to save
    :param name: name of the training
    :param loss: the loss at this point
    :return: None
    """
    # delete existing weights file
    files = glob.glob(os.path.join(name, 'weights_' + '*' + '.h5'))
    for file in files:
        os.remove(file)
    path_name = os.path.join(name, 'weights_' + str(loss) + '.h5')
    model.save(path_name)


def get_dataset(color_path, depth_path, label_path, batch_size, image_h=480, image_w=640, shuffle=True):
    """
    Fucntion to create a tensorflow dataset API pipeline to stream data in batches
    :param color_path: list of all the color paths
    :param depth_path: list of all the depth paths
    :param label_path: list of all the label paths
    :param batch_size: Batch size to return the data in
    :param image_h: Height to resize the images to
    :param image_w: Width to resize the images to
    :param shuffle: Whether to shuffle the data or not
    :return: the prefetched dataset object
    """
    dataset = tf.data.Dataset.from_tensor_slices((color_path, depth_path, label_path))
    if shuffle:
        dataset = dataset.shuffle(len(color_path))
    dataset = dataset.map(lambda c, d, l: read_images(c, d, l, image_h, image_w),
                          num_parallel_calls=10)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(5)
    print('-------------------')
    print('Dataset:')
    print('Images count: {}'.format(len(color_path)))
    print('Step per epoch: {}'.format(len(color_path) // batch_size))
    print('Images per epoch: {}'.format(batch_size * (len(color_path) // batch_size)))
    return dataset


def dataset_generator(dataset):
    """
    Fucntion to create a python iterator to stream data
    :param dataset: Prefetched dataset API object
    :return: Decoded colpr images, Decoded depth images, decodes label converted from RGB to G
    """
    for batch in dataset:
        labels = tf.where(batch[2][:, :, :, 1] > 0.0, 1, 0)
        yield batch[0], batch[1], tf.expand_dims(labels, -1)


def dataset_generator_predict(dataset):
    """
    Fucntion to create a python iterator to stream data
    :param dataset: Prefetched dataset API object
    :return: Decoded colpr images, Decoded depth images, decodes label converted from RGB to G
    """
    for batch in dataset:
        yield batch


def model(image_h, image_w):
    """
    Fucntion to define the encoder decoder architecture for segmentation
    I use FCN-8 architecture with VGG16 as the encoder and pretrained on imagenet
    :param image_h: Height of the input image
    :param image_w: Width of the input image
    :return: A keras model object
    """
    input_image = tf.keras.layers.Input((image_h, image_w, 3), dtype='float32')
    vgg = VGG16(include_top=False,
                weights='imagenet',
                input_tensor=input_image,
                input_shape=(image_h, image_w, 3),
                pooling=None)

    pool_3 = vgg.get_layer('block3_pool').output
    pool_4 = vgg.get_layer('block4_pool').output

    x = Conv2D(4096, (2, 2), padding="valid", activation="relu", name="fc6")(vgg.output)
    x = Conv2D(4096, (1, 1), padding="valid", activation="relu", name="fc7")(x)
    encoder_out = Conv2D(2, (1, 1), padding="valid", activation="relu", name="encoder_output")(x)
    # input_graph, pool_3, pool_4, encoder_out

    score_2 = Conv2DTranspose(2, 4, strides=(2, 2), padding='valid')(encoder_out)
    score_pool_4 = Conv2D(2, 1, padding='valid', use_bias=True)(pool_4)
    # score_pool_4 = Cropping2D(cropping=(1,0))(score_pool_4)
    score_2 = ZeroPadding2D(padding=((1, 0), (0, 0)))(score_2)

    score_16x_upsampled = Add()([score_2, score_pool_4])

    # Unpool to 8x
    score_4 = Conv2DTranspose(2, 4, strides=(2, 2), padding='valid')(score_16x_upsampled)
    score_pool_3 = Conv2D(2, 1, padding='valid', use_bias=True)(pool_3)
    score_pool_3 = ZeroPadding2D(padding=((1, 1), (1, 1)))(score_pool_3)
    # score_pool_3 = Cropping2D(cropping=9)(score_pool_3)
    score_8x_upsampled = Add()([score_4, score_pool_3])

    # Unpool to image shape
    upsample = Conv2DTranspose(2, 16, strides=(8, 8), padding='same')(score_8x_upsampled)
    upsample = Cropping2D(cropping=(8, 8))(upsample)

    # output = Activation('softmax')(upsample)
    final_model = tf.keras.models.Model(input_image, upsample)
    # tf.keras.utils.plot_model(final_model, 'model.png', show_shapes=True)
    return final_model


def grad(model, color, label, training=True):
    """
    Fucntion to run the model on the given image and compute gradient and weifhted loss
    :param model: Keras model to be trained
    :param color: Batch of color images
    :param label: Batch to labels
    :param training: Whether training to validation phase
    :return: weighted loss, model prediction without softmax, correct one hot coded labels and gradients
    """
    with tf.GradientTape() as tape:
        y_pred = model(color, training)
        logits = tf.reshape(y_pred, (-1, 2))
        correct_label = tf.one_hot(np.squeeze(tf.reshape(label, (-1, 1)).numpy()), 2)
        # define loss function
        weights = tf.reduce_sum([1.0, 3.0] * correct_label, axis=1)
        unweighted_cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
        weighted_cross_entropy_loss = tf.reduce_mean(unweighted_cross_entropy_loss * weights)
    if training:
        return weighted_cross_entropy_loss, logits, correct_label, tape.gradient(weighted_cross_entropy_loss,
                                                                                 model.trainable_variables)
    elif not training:
        return weighted_cross_entropy_loss, logits, correct_label


def train(epochs, model, train_dataset, val_dataset, steps_per_epoch_train, steps_per_epoch_val, train_name='train'):
    """
    Fucntion to train the model to predict the falt surface for suction
    :param epochs: Number of time you want to run the training to whole data
    :param model: the Keras model to be trained
    :param train_dataset: the iterator to stream batch of training data
    :param val_dataset: the iterator to stram batch of validation data
    :param steps_per_epoch_train: Number to batches to train in one epoch
    :param steps_per_epoch_val: Number of bacthed to validate in one epoch
    :param train_name: Name of the training
    :return: None
    """

    path = "./logs/"
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=False, onerror=None)
    os.makedirs(path)

    tf.keras.utils.plot_model(model, os.path.join(path, 'FCN8.png'), show_shapes=True)

    num_epochs = epochs
    steps_per_epoch_train = steps_per_epoch_train
    steps_per_epoch_val = steps_per_epoch_val

    train_loss = tf.keras.metrics.Mean()
    train_accracy = tf.keras.metrics.Accuracy()
    train_precision = tf.keras.metrics.Precision()

    val_loss = tf.keras.metrics.Mean()
    val_accracy = tf.keras.metrics.Accuracy()
    val_precision = tf.keras.metrics.Precision()

    best_val_loss = 1e6

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # log (tensorboard)
    summary_writer_train = tf.summary.create_file_writer(os.path.join(path, 'train'), flush_millis=20000)
    summary_writer_val = tf.summary.create_file_writer(os.path.join(path, 'val'), flush_millis=20000)

    # training
    for epoch in range(num_epochs):

        pbar_train = tqdm(range(steps_per_epoch_train))
        for _ in pbar_train:
            color, depth, label = next(train_dataset)

            total_loss, logits, correct_label, grads = grad(model, color, label, training=True)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss.update_state(total_loss)
            train_accracy.update_state(tf.argmax(correct_label, axis=-1), tf.argmax(logits, axis=-1))
            train_precision.update_state(tf.argmax(correct_label, axis=-1), tf.argmax(logits, axis=-1))
            pbar_train.set_description(
                "Train_Epoch {}, Train_loss {:.4f}".format(epoch, train_loss.result().numpy()))
        log_loss(train_loss, train_accracy, train_precision, epoch, summary_writer_train)

        pbar_val = tqdm(range(steps_per_epoch_val))
        for _ in pbar_val:
            color, depth, label = next(val_dataset)
            total_loss, logits, correct_label = grad(model, color, label, training=False)
            val_loss.update_state(total_loss)
            val_accracy.update_state(tf.argmax(correct_label, axis=-1), tf.argmax(logits, axis=-1))
            val_precision.update_state(tf.argmax(correct_label, axis=-1), tf.argmax(logits, axis=-1))

            pbar_val.set_description(
                "Val_Epoch {}, Val_loss {:.4f}".format(epoch, val_loss.result().numpy()))
        log_loss(val_loss, val_accracy, val_precision, epoch, summary_writer_val)

        # save
        if val_loss.result().numpy() < best_val_loss:
            save_best_weights(model, path, val_loss.result().numpy())
            best_val_loss = val_loss.result().numpy()

        train_loss.reset_states()
        train_accracy.reset_states()
        train_precision.reset_states()

        val_loss.reset_states()
        val_accracy.reset_states()
        val_precision.reset_states()

    return None


def main(mode='train', pretraind_path=None):
    """
    Main function to train, validate and predict on the data given
    :param mode: 'train' or 'predict'
    :param pretraind_path: in case of predict the path to the trained saved model
    :return: None
    """

    image_h = 240
    image_w = 320

    if mode == 'predict':
        if pretraind_path is None or not os.path.exists(pretraind_path):
            print("Path does not exits")
            exit(0)
        else:
            test_data = "../data/test/color/"
            test_data_path = os.listdir(test_data)
            dataset = tf.data.Dataset.from_tensor_slices((test_data_path))
            dataset = dataset.map(lambda c: tf_read_image(test_data + c, image_h, image_w, channels=0, norm=True),
                                  num_parallel_calls=10).repeat(1).batch(1)

            gen = dataset_generator_predict(dataset)
            final_model = tf.keras.models.load_model(pretraind_path)

            for i, image in enumerate(gen):
                output = final_model(image, training=False)
                output = tf.math.argmax(output[0], 2)
                final_op = np.zeros((image_h, image_w, 3))
                final_op[:, :, 1] = output.numpy() * 255.0
                cv2.imwrite("output/" + test_data_path[i], final_op)
            print("Exiting Code , prediction finished")

    elif mode == 'train':

        train_data = "../data/train/train_data.txt"
        val_data = "../data/train/val_data.txt"
        epochs = 100
        batch_size = 8

        train_data_color_path, train_data_depth_path, train_data_label_path, val_data_color_path, val_data_depth_path, \
        val_data_label_path = create_data(train_data, val_data)
        train_dataset = get_dataset(train_data_color_path, train_data_depth_path, train_data_label_path, batch_size,
                                    image_h, image_w)
        val_dataset = get_dataset(val_data_color_path, val_data_depth_path, val_data_label_path, batch_size, image_h,
                                  image_w)
        train_gen = dataset_generator(train_dataset)
        val_gen = dataset_generator(val_dataset)
        final_model = model(image_h=image_h, image_w=image_w)
        # tf.keras.utils.plot_model(final_model, 'model.png', show_shapes=True)
        train(epochs, final_model, train_gen, val_gen, len(train_data_color_path) // batch_size,
              len(val_data_color_path) // batch_size, "train_1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is a program to train suction challenge ')
    parser.add_argument('--mode', default='predict')
    parser.add_argument('--model_path', default='./logs/weights_0.07724742.h5')
    args = parser.parse_args()
    main(mode=args.mode, pretraind_path=args.model_path)
