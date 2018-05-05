from matplotlib import pyplot as plt
from vis_utils import visualize_grid
import numpy as np


def show_traning(loss_history, train_acc_history, val_acc_history):
    plt.subplot(2, 1, 1)
    plt.plot(loss_history, 'o')
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.subplot(2, 1, 2)
    plt.plot(train_acc_history, '-o')
    plt.plot(val_acc_history, '-o')
    plt.legend(['train', 'val'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    plt.show()


def compare_trainings(loss_train_val_acc_with_labels):
    plt.subplot(3, 1, 1)
    for label in loss_train_val_acc_with_labels:
        plt.plot(loss_train_val_acc_with_labels[label][0], 'o', label=label)
    plt.title('Train loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend(ncol=2, loc='lower right')
  
    plt.subplot(3, 1, 2)
    for label in loss_train_val_acc_with_labels:
        plt.plot(loss_train_val_acc_with_labels[label][1], '-o', label=label)
    plt.title('Train accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(ncol=2, loc='lower right')
    
    plt.subplot(3, 1, 3)
    for label in loss_train_val_acc_with_labels:
        plt.plot(loss_train_val_acc_with_labels[label][2], '-o', label=label)
    plt.title('Val accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(ncol=2, loc='lower right')
    
    plt.gcf().set_size_inches(15, 15)
    plt.show()


def plot_mean_std_hist(parameters):
    ws = parameters

    ws_means = [np.mean(w) for w in ws]
    ws_stds = [np.std(w) for w in ws]
    
    for i in range(len(ws)):
        print('weight index: ', i)
        print('with size ', ws[i].shape)
        print('with mean: %f, std: %f' % (ws_means[i], ws_stds[i]))
        
    plt.figure()
    plt.subplot(121)
    plt.plot(range(len(ws_means)), ws_means, 'ob-')
    plt.title('weights means')
    plt.subplot(122)
    plt.plot(range(len(ws_stds)), ws_stds, 'or-')
    plt.title('weights stds')

    plt.figure()
    for i in range(len(ws)):
        plt.subplot(len(ws), 1, i+1)
        #plt.subplot(1, len(ws), i+1)
        plt.hist(ws[i].ravel(), 30)
        plt.title('weights %d hist' % i)
    
    plt.show()


def show_conv_weights(conv_weights):
    grid = visualize_grid(conv_weights)
    plt.imshow(grid.astype('uint8'))
    plt.axis('off')
    plt.gcf().set_size_inches(5, 5)
    plt.show()
