import timeit
import numpy as np
import cv2
import os
from os.path import join
from skimage.feature import hog
import random
from sklearn.cluster import KMeans
from sklearn import svm
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


def load_images_training():
    images_dict = {}
    label = []
    cntr = 0
    # os.walk will access all the sub-folders itself from scene_categories
    path = '...scene_categories' #add data folder path here
    for root, dirs, files in os.walk(path, topdown=True):
        x = 0
        label.append(cntr)
        cntr += 1
        category = []
        print('Class changed!')
        for names in files:
            x += 1
            if x <= 50:
                # Pyramid level 1
                img = cv2.imread(join(root, names))
                cv2.imshow('Train1 img', img)
                cv2.waitKey(10)

                # Pyramid level 2
                img_scale = cv2.pyrUp(img)  # Scale ratio is 2.0
                cv2.imshow('Train2 img', img_scale)
                cv2.waitKey(10)

                # Pyramid level 3
                img_scale_2 = cv2.pyrUp(img_scale)
                cv2.imshow('Train3 img', img_scale_2)
                cv2.waitKey(10)

                category.append(img)
                category.append(img_scale)
                category.append(img_scale_2)

        images_dict[root] = category

    label_nd = np.asarray(label)
    return [images_dict, label_nd]


def load_images_test():
    images_dict = {}
    label = []
    cntr = 0
    # os.walk will access all the sub-folders itself from scene_categories
    path = '...scene_categories' #add data folder path here
    for root, dirs, files in os.walk(path, topdown=True):
        x = 0
        label.append(cntr)
        cntr += 1
        category = []
        print('Class changed!')
        for names in files:
            x += 1
            if x <= 200:
                pass
            elif x >= 200 and x <= 210:
                # Pyramid level 1
                img = cv2.imread(join(root, names))

                # Pyramid level 2
                img_scale = cv2.pyrUp(img)  # Scale ratio is 2.0

                # Pyramid level 3
                img_scale_2 = cv2.pyrUp(img_scale)

                category.append(img)
                category.append(img_scale)
                category.append(img_scale_2)

        images_dict[root] = category

    label_nd = np.asarray(label)
    return [images_dict, label_nd]


def features(images_dict):
    features_dict = {}
    descriptors = []
    for key, value in images_dict.items():
        feature_dp = []
        for img in value:
            hog_dp, hog_img = hog(img, orientations=5, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                              transform_sqrt=True, visualize=True, multichannel=True)

            # Randomly selecting 1000 descriptors from the image
            hog_lst = list(hog_dp)
            hog_lst = random.sample(hog_lst, 1000)
            hog_rnd = np.asarray(hog_lst)

            # Reshaping
            hog_rnd = hog_rnd.reshape((len(hog_rnd), 1))
            hog_dp_32 = np.float32(hog_rnd)

            # Storing descriptors
            descriptors.extend(hog_dp_32)
            feature_dp.append(hog_dp_32)
        features_dict[key] = feature_dp

    return [descriptors, features_dict]


def clustering(descriptors):
    cltr = KMeans(n_clusters=50, n_init=10)
    cltr.fit(descriptors)
    visual_words = cltr.cluster_centers_

    return visual_words


def indexing(img, center):
    cntr = 0
    indx = 0
    for i in range(len(center)):
        if i == 0:
            cntr = distance.euclidean(img, center[i])
        else:
            dist = distance.euclidean(img, center[i])
            if dist < cntr:
                indx = i
                cntr = dist
    return indx


def histograms(class_dict, centroids):
    hist_dict = {}
    hist_list = []
    labels = []
    for key, value in class_dict.items():
        each_class = []
        for img in value:
            hist_freq = np.zeros(len(centroids))
            for each_feature in img:
                indx = indexing(each_feature, centroids)
                hist_freq[indx] += 1  # frequency of features
            each_class.append(hist_freq)
        hist_dict[key] = each_class

    for key, val in hist_dict.items():
        for item in val:
            hist_list.append(item)
            labels.append(key)

    return [hist_list, labels]


def main():
    start = timeit.default_timer()

    print()
    print('------TRAINING PHASE------')
    print()
    start_load_train = timeit.default_timer()
    print('Loading training images')
    images, labels = load_images_training()
    print('Images loaded')
    end_load_train = timeit.default_timer()
    end_load_train_time = end_load_train - start_load_train
    print('Run time: {} seconds'.format(round(end_load_train_time, 2)))

    print()
    start_features_train = timeit.default_timer()
    print('Extracting descriptors')
    descriptors, features_dict = features(images)
    print('Descriptors extracted')
    end_features_train = timeit.default_timer()
    end_features_train_time = end_features_train - start_features_train
    print('Run time: {} seconds'.format(round(end_features_train_time, 2)))

    print()
    start_cluster_train = timeit.default_timer()
    print('Performing K-Means clustering')
    centroids = clustering(descriptors)
    print('K-Means completed')
    end_cluster_train = timeit.default_timer()
    end_cluster_train_time = end_cluster_train - start_cluster_train
    print('Run time: {} seconds'.format(round(end_cluster_train_time, 2)))

    print()
    start_hist_train = timeit.default_timer()
    print('Generating histograms of training data')
    hist_list, hist_labels = histograms(features_dict, centroids)
    print('Processing complete')
    end_hist_train = timeit.default_timer()
    end_hist_train_time = end_hist_train - start_hist_train
    print('Run time: {} seconds'.format(round(end_hist_train_time, 2)))

    # SVM
    print()
    start_svm_train = timeit.default_timer()
    print('Performing SVM')
    print('length of list', len(hist_list))
    print('length of labels', len(hist_labels))
    clf = svm.SVC(kernel='rbf', cache_size=500, random_state=1)
    clf.fit(hist_list, hist_labels)
    print('SVM trained successfully')
    end_svm_train = timeit.default_timer()
    end_svm_train_time = end_svm_train - start_svm_train
    print('Run time: {} seconds'.format(round(end_svm_train_time, 2)))

    print()
    print('------TEST PHASE------')
    print()
    start_load_test = timeit.default_timer()
    print('Loading test images')
    images_test, labels_test = load_images_test()
    print('Images loaded')
    end_load_test = timeit.default_timer()
    end_load_test_time = end_load_test - start_load_test
    print('Run time: {} seconds'.format(round(end_load_test_time, 2)))

    print()
    start_features_test = timeit.default_timer()
    print('Extracting features of test images')
    descriptors_test, features_dict_test = features(images_test)
    print('Test images features extracted')
    end_features_test = timeit.default_timer()
    end_features_test_time = end_features_test - start_features_test
    print('Run time: {} seconds'.format(round(end_features_test_time, 2)))

    print()
    start_hist_test = timeit.default_timer()
    print('Generating histograms of test data')
    hist_list_test, hist_labels_test = histograms(features_dict_test, centroids)
    print('Processing complete')
    end_hist_test = timeit.default_timer()
    end_hist_test_time = end_hist_test - start_hist_test
    print('Run time: {} seconds'.format(round(end_hist_test_time, 2)))

    print()
    start_predict_test = timeit.default_timer()
    print('Predicting test data')
    # print('hist_list_test: ', hist_list_test)
    print('hist_list_test length: ', len(hist_list_test))
    predicted = clf.predict(hist_list_test)
    # print('Predict test: ', predicted)
    print('Predict test length: ', len(predicted))
    end_predict_test = timeit.default_timer()
    end_predict_test_time = end_predict_test - start_predict_test
    print('Run time: {} seconds'.format(round(end_predict_test_time, 2)))

    print()
    accuracy = accuracy_score(hist_labels_test, predicted)
    print('Test Accuracy: ', accuracy)

    print()
    print('Confusion matrix')
    conf_mat = confusion_matrix(hist_labels_test, predicted)
    categories = ['Bedroom', 'CAL Suburb', 'Industrial', 'Kitchen', 'Living Room', 'MIT Coast',
                  'MIT Forest', 'MIT Highway', 'MIT Inside City', 'MIT Mountain', 'MIT Open Country',
                  'MIT Street', 'MIT Tall Building', 'PAR Office', 'Store']

    print(conf_mat)
    title = 'Confusion matrix'
    fig, ax_mat = plt.subplots()
    im = ax_mat.imshow(conf_mat, cmap=plt.cm.Blues)
    plt.xticks(np.arange(len(conf_mat)), categories)
    plt.xticks(rotation=90)
    plt.yticks(np.arange(len(conf_mat)), categories)
    for i in range(len(conf_mat)):
        for j in range(len(conf_mat)):
           text = ax_mat.text(j, i, conf_mat[i, j])

    ax_mat.set_title(title)
    fig.tight_layout()

    print()
    end = timeit.default_timer()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Run time: {:0>2} hrs {:0>2} mins {:05.2f} sec".format(int(hours), int(minutes), seconds))

    plt.show()


if __name__ == "__main__":main()