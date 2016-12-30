import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt
from neural_net import TwoLayerNet
import cPickle, gzip



def main(argv=None):
    # Load the dataset
    try:
        print 'Loading the dataset...'
        f = gzip.open('DATA/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
    except:
        print 'ERROR!'
        print 'You must first download the data.'
        print "Run the 'get_mnist_data.sh' script in the DATA directory."
        return

    def rel_error(x,y):
        """ returns relative error """
        return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

    # setup training, validation, and test sets
    X_train = np.array(train_set[0])
    y_train = np.array(train_set[1])
    X_val = np.array(valid_set[0])
    y_val = np.array(valid_set[1])
    X_test = np.array(test_set[0])
    y_test = np.array(test_set[1])

    input_size = 28 * 28    # images are 28 x 28 pixels
    hidden_size = 100       # number of nodes in hidden layer
    num_classes = 10        # number of classes: 0, 1, ..., 9

    # initialize the network
    net = TwoLayerNet(input_size, hidden_size, num_classes)

    # train the network
    print 'Training the neural network...'

    stats = net.train(X_train, y_train, X_val, y_val,
                        num_iters=1000, batch_size=200,
                        learning_rate=0.5, learning_rate_decay=0.95,
                        reg = 1e-4, verbose=True)

    # predict on the validation set
    print 'Running the network on the validation set...'
    val_acc = (net.predict(X_val) == y_val).mean()
    print 'Validation accuracy: ', val_acc
    print ''

    # predict on the test set
    print 'Running the network on the test set...'
    test_acc = (net.predict(X_test) == y_test).mean()
    print 'Test accuracy: ', test_acc
    print ''


    # plot the loss function and train / validation accuracies
    fig1 = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2,1,2)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')
    plt.legend(loc='lower right')

    plt.show()
    print 'Creating plots of the loss and classification accuracy...'
    fig1.savefig('loss_classification_accuracy.png')
    print "Plot written to loss_classification_accuracy.png"
    print ''


if __name__ == "__main__":
    sys.exit(main())

