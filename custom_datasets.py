# custom_datasets.py
import tensorflow as tf

class PyDataset(tf.data.Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Properly call superclass init
       


