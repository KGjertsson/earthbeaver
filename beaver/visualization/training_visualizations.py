import matplotlib.pyplot as plt
import numpy as np


# Visualize accuracies
def plot_training_history(history, what='loss'):
    x = history.history[what]
    val_x = history.history['val_' + what]
    epochs = np.asarray(history.epoch) + 1

    plt.plot(epochs, x, label="Training " + what)
    plt.plot(epochs, val_x, label="Validation " + what)
    plt.title("Training and validation " + what)
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
    return None
