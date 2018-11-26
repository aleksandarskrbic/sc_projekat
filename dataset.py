import os
import cv2
import time
import numpy as np
from multiprocessing import Process


train_data_path = './imgs/train'
test_data_path = './imgs/test'
val_data_path = './imgs/val'
img_shape = (128, 128)


def timeit(method):
    def timed(*args, **kw):
        start = time.time()
        result = method(*args, **kw)
        end = time.time()
        print(f'{method.__name__} execution time => {(end - start):.2f}')
        return result
    return timed


@timeit
def create_data(path):
    imgs = []
    labels = []

    if path == train_data_path:
        print('Creating training data...')
    elif path == test_data_path:
        print('Creating test data...')
    else:
        print('Creating validation data...')

    for img_name in os.listdir(os.path.join(path, 'NORMAL')):
        img_path = path + '/NORMAL/' + img_name
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, img_shape)
        imgs.append(img)
        labels.append(0)

    for img_name in os.listdir(os.path.join(path, 'PNEUMONIA')):
        img_path = path + '/PNEUMONIA/' + img_name
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, img_shape)
        imgs.append(img)
        labels.append(1)

    x = np.array(imgs)
    y = np.array(labels)

    if path == train_data_path:
        np.save('./data/x_train.npy', x)
        np.save('./data/y_train.npy', y)
    elif path == test_data_path:
        np.save('./data/x_test.npy', x)
        np.save('./data/y_test.npy', y)
    else:
        np.save('./data/x_val.npy', x)
        np.save('./data/y_val.npy', y)


if __name__ == '__main__':
    train_process = Process(target=create_data, args=(train_data_path,))
    train_process.start()

    test_process = Process(target=create_data, args=(test_data_path,))
    test_process.start()

    val_process = Process(target=create_data, args=(val_data_path,))
    val_process.start()

    train_process.join()
    test_process.join()
    val_process.join()
