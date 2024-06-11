import matplotlib.pyplot as plt

def visualize_sample(sample: tuple, prediction=None):
    if prediction is not None:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(15, 15))
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(15, 15))

    sparse, rgb, gt = sample

    ax1.set_title("RGB")
    ax1.imshow(rgb.transpose(1, 2, 0), interpolation='nearest')

    ax2.set_title("Sparse")
    ax2.imshow(sparse.transpose(1, 2, 0), interpolation='nearest', label="Sparse image")

    ax3.set_title("Groundtruth")
    ax3.imshow(gt.transpose(1, 2, 0),interpolation='nearest')

    if prediction is not None:
        ax4.set_title("Predicted")
        ax4.imshow(prediction.transpose(1, 2, 0), interpolation='nearest')

    plt.show()