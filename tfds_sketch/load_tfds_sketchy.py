import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import tfds_sketchy

def view_pair_dataset(ds, n_rows) :    
    _, ax = plt.subplots(n_rows, 2)
    for i in range(n_rows) :
        for j in range(2) :
            ax[i,j].set_axis_off()

    ds = next(ds.as_numpy_iterator())
    photos = ds['photo']
    sketches = ds['sketch']
    for i, (sketch, photo) in enumerate(zip(sketches, photos)):
        ax[i][0].imshow(sketch)
        ax[i][1].imshow(photo)
    plt.show()

# Load your custom dataset
train_ds, test_ds = tfds.load(
    'tfds_sketchy',
    split=['train', 'test']
    )

batch_size = 6
train_ds = train_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)


view_pair_dataset(train_ds, batch_size)
view_pair_dataset(test_ds, batch_size)


input()
