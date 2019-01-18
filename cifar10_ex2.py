import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile

cifar10_dataset_folder_path = 'cifar-10-batches-py'

history = {'val_loss': [],
               'val_acc': []}

class DownloadProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

#데이터 라벨별 이름
def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#배치데이터(1,2,3,4,5) 사이즈 32x32x3으로 변경
def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

#이미지 보기
def display_stats(cifar10_dataset_folder_path, batch_id, sample_id):
    features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)

    if not (0 <= sample_id < len(features)):
        print('{} samples in batch {}.  {} is out of range.'.format(len(features), batch_id, sample_id))
        return None

    print('\nStats of batch #{}:'.format(batch_id))
    print('# of Samples: {}\n'.format(len(features)))

    #라벨(0~9)이름
    label_names = load_label_names()
    #라벨(0~9)개수
    label_counts = dict(zip(*np.unique(labels, return_counts=True)))
    for key, value in label_counts.items():
        print('Label Counts of [{}]({}) : {}'.format(key, label_names[key].upper(), value))

    sample_image = features[sample_id]
    sample_label = labels[sample_id]

    print('\nExample of Image {}:'.format(sample_id))
    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))

#min-max 정규화 0~1
def normalize(x):
    min = np.min(x)
    max = np.max(x)

    x = (x-min)/(max-min)
    return x

#원 핫 인코딩 구현
def one_hot_encoding(x):
    encoded = np.zeros((len(x),10))

    for index,value in enumerate(x):
        encoded[index][value] = 1

    return encoded

#90%-trainset 10%-vaildationset
def preprocess_and_save(normalize,one_hot_encoding,features,labels,filename):
    features = normalize(features)
    labels = one_hot_encoding(labels)

    pickle.dump((features,labels),open(filename,'wb'))

def preprocess_and_save_data(cifar10_dataset_folder_path,normalize,one_hot_encoding):
    batchs = 5
    valid_features = []
    valid_labels = []

    for batch_id in range(1,batchs+1):
        features,labels = load_cfar10_batch(cifar10_dataset_folder_path,batch_id)

        #10%-validation
        index_of_validation = int(len(features) * 0.1)

        #save-train-batch 90%
        preprocess_and_save(normalize,one_hot_encoding,
                            features[:-index_of_validation],labels[:-index_of_validation],
                            'preprocess_batch_' + str(batch_id) + '.p')

        # save-validation 10%
        valid_features.extend(features[-index_of_validation:])
        valid_labels.extend(labels[-index_of_validation:])

        preprocess_and_save(normalize, one_hot_encoding,
                            np.array(valid_features), np.array(valid_labels),
                            'preprocess_validation.p')

        #save-test
        with open(cifar10_dataset_folder_path+'/test_batch',mode='rb') as file:
            batch = pickle.load(file,encoding='latin1')

        test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        test_labels = batch['labels']

        preprocess_and_save(normalize, one_hot_encoding,
                             np.array(test_features), np.array(test_labels),
                             'preprocess_training.p')

def cnn(x,keep_prob):
    #he-initializer
    initializer = tf.contrib.layers.variance_scaling_initializer()

    W1 = tf.Variable(initializer([3,3,3,64]))
    W2 = tf.Variable(initializer([3,3,64,128]))
    W3 = tf.Variable(initializer([5,5,128,256]))
    W4 = tf.Variable(initializer([5,5,256,512]))

    '''
    W1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08))
    W2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
    W3 = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 256], mean=0, stddev=0.08))
    W4 = tf.Variable(tf.truncated_normal(shape=[5, 5, 256, 512], mean=0, stddev=0.08))
    '''
    '''
    b1 = tf.Variable(tf.constant(0.0005,shape=[64]))
    b2 = tf.Variable(tf.constant(0.0005,shape=[128]))
    b3 = tf.Variable(tf.constant(0.0005,shape=[256]))
    b4 = tf.Variable(tf.constant(0.0005,shape=[512]))
    '''
    conv1 = tf.nn.conv2d(x,W1,strides=[1,1,1,1],padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    conv2 = tf.nn.conv2d(conv1_bn,W2,strides=[1,1,1,1],padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv2_bn = tf.layers.batch_normalization(conv2_pool)

    conv3 = tf.nn.conv2d(conv2_bn,W3,strides=[1,1,1,1],padding='SAME')
    conv3 = tf.nn.relu(conv3)
    conv3_pool = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv3_bn = tf.layers.batch_normalization(conv3_pool)

    conv4 = tf.nn.conv2d(conv3_bn,W4,strides=[1,1,1,1],padding='SAME')
    conv4 = tf.nn.relu(conv4)
    conv4_pool = tf.nn.max_pool(conv4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv4_bn = tf.layers.batch_normalization(conv4_pool)

    flat = tf.contrib.layers.flatten(conv4_bn)

    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=256, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob)
    full1 = tf.layers.batch_normalization(full1)

    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
    full2 = tf.nn.dropout(full2, keep_prob)
    full2 = tf.layers.batch_normalization(full2)

    full3 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=512, activation_fn=tf.nn.relu)
    full3 = tf.nn.dropout(full3, keep_prob)
    full3 = tf.layers.batch_normalization(full3)

    full4 = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=1024, activation_fn=tf.nn.relu)
    full4 = tf.nn.dropout(full4, keep_prob)
    full4 = tf.layers.batch_normalization(full4)

    out = tf.contrib.layers.fully_connected(inputs=full4,num_outputs=10,activation_fn = None)

    return out

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    session.run(optimizer,
                feed_dict={
                    x: feature_batch,
                    y: label_batch,
                    keep_prob: keep_probability
                })


def print_stats(session, feature_batch, label_batch, cost, accuracy):
    loss = sess.run(cost,
                    feed_dict={
                        x: feature_batch,
                        y: label_batch,
                        keep_prob: 1.
                    })
    valid_acc = sess.run(accuracy,
                         feed_dict={
                             x: valid_features,
                             y: valid_labels,
                             keep_prob: 1.
                         })

    history['val_loss'].append(loss)
    history['val_acc'].append(valid_acc * 100)

    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))

def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

def load_preprocess_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)


if not isfile('cifar-10-python.tar.gz'):
    with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            'cifar-10-python.tar.gz',
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open('cifar-10-python.tar.gz') as tar:
        tar.extractall()
        tar.close()

preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encoding)

valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
y =  tf.placeholder(tf.float32, shape=(None, 10), name='output_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

epochs = 10000
batch_size = 128
keep_probability = 0.7
learning_rate = 0.001

logits = cnn(x, keep_prob)
model = tf.identity(logits, name='logits') # Name logits Tensor, so that can be loaded from disk after training

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)

            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)

    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)

# 도화지 생성
fig = plt.figure()
# 정확도 그래프 그리기
plt.plot(range(epochs*5), history['val_acc'], label='Accuracy', color='darkred')
# 축 이름
plt.xlabel('epochs')
plt.ylabel('검증 정확도(%)')
plt.title('Graph')
plt.grid(linestyle='--', color='lavender')
# 그래프 표시
plt.show()
plt.savefig('mnist_tensorflow_acc.png')