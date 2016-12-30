import numpy as np
import matplotlib.pyplot as plt

from neural_net import TwoLayerNet
from vis_utils import visualize_grid
from load_data import get_CIFAR10_data

input_size = 32 * 32 * 3
hidden_size = 75
num_classes = 10
best_net = TwoLayerNet(input_size, hidden_size, num_classes)

# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

# Train the network
stats = best_net.train(X_train, y_train, X_val, y_val,
            num_iters=1000, batch_size=200,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=0.5, verbose=True)

# Predict on the validation set
val_acc = (best_net.predict(X_val) == y_val).mean()
print 'Validation accuracy: ', val_acc


fig1 = plt.figure(1)
# Plot the loss function and train / validation accuracies
ax1 = plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

ax2 = plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.show()
fig1.savefig("classification_accuracy.png")

# Visualize the weights of the network

def show_net_weights(net):
  W1 = net.params['W1']
  W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
  plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
  plt.gca().axis('off')
  plt.show()

fig2 = plt.figure(2)
show_net_weights(net)
fig2.savefig("learned_weights.png")
