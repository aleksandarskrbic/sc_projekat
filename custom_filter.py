from multiprocessing import Process
import numpy as np


def get_feature_vector(source):
    features = []
    for y in range(1, source.shape[0]-1):
        for x in range(1, source.shape[1]-1):
            members = []
            members.append(source[y-1, x-1])
            members.append(source[y, x-1])
            members.append(source[y+1, x-1])
            members.append(source[y-1, x])
            members.append(source[y, x])
            members.append(source[y+1, x])
            members.append(source[y-1, x+1])
            members.append(source[y, x+1])
            members.append(source[y+1, x+1])
            features.append(sum(members)/9)
    features = np.asarray(features)
    return features


def calculate_features(imgs, i):
    features = []
    for img in imgs:
        features.append(get_feature_vector(img))

    features = np.array(features)
    np.save(f'./data/features_{i}', features)


if __name__ == '__main__':
    x_train = np.load('./data/x_train.npy')
    x_test = np.load('./data/x_test.npy')
    x = np.vstack((x_train, x_test))
    chunks = np.array_split(x, 4)

    processes = []
    for i in range(4):
        p = Process(target=calculate_features, args=(chunks[i], i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
