import tensorflow as tf
from Loss import PixWiseBCELoss
import numpy as np

class Model:
    def __init__(self) -> None:
        # Define the 'enc' variable as described
        dense = tf.keras.applications.DenseNet121(include_top=False)
        self.features = dense.layers
        # Extract desirable layers from DenseNet121 
        enc = self.features[:109]
        x = enc[-1].output  
        out_feature_map = tf.keras.layers.Conv2D(1,(1,1), padding='valid', strides=1, activation='sigmoid')(x)
        out_map_flat = tf.reshape(out_feature_map, (-1, 28 * 28))
        out_binary = tf.keras.layers.Dense(1, activation='sigmoid')(out_map_flat)
        input = enc[0].input
        self.model = tf.keras.Model(inputs=input, outputs=[out_feature_map, out_binary])
        self.loss = PixWiseBCELoss(beta=0.5)
        self.optimizer = tf.keras.optimizers.Adam()
    
    # train model    
    def fit(self, x_train, y_train_label, y_train_mask, epochs=100, batch_size=32, save_best_weights=True, shuffle=True):
        if shuffle == True:
            # shuffle data
            # we shuflled data first
            indices = np.arange(x_train.shape[0])
            # we shuffled indices
            np.random.shuffle(indices)
            # reassign data(features dataset) and y dataset
            x_train = x_train[indices]
            y_train_label = y_train_label[indices]
            y_train_mask = y_train_mask[indices]
        # using GradientTape to update weights
        @tf.function
        def train_step(x_batch, y_mask_batch, y_label_batch):
            with tf.GradientTape() as tape:
                out_mask, out_label = self.model(x_batch)
                batch_loss = self.loss([y_mask_batch, y_label_batch], [out_mask, out_label])

            gradients = tape.gradient(batch_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            return batch_loss

        # Training loop
        epochs = epochs
        epochs_losses = []
        batch_size = batch_size
        num_batches = len(x_train) // batch_size

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            epoch_loss = 0

            for batch in range(num_batches):
                start = batch * batch_size
                end = (batch + 1) * batch_size

                x_batch = x_train[start:end]
                y_mask_batch = y_train_mask[start:end]
                y_label_batch = y_train_label[start:end]

                batch_loss = train_step(x_batch, y_mask_batch, y_label_batch)
                epoch_loss += batch_loss.numpy()

            scaled_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} Loss: {scaled_loss}")
            if save_best_weights:
                epochs_losses.append(scaled_loss)
                # save best weights of model
                if  scaled_loss <= min(epochs_losses):
                    print(f'Weights of Model saved with Loss: {scaled_loss}')
                    self.model.save_weights('Weights/best_weights.h5')
        

