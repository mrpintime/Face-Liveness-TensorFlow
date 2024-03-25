import tensorflow as tf

# Create Loss function
class PixWiseBCELoss(tf.keras.losses.Loss):
    def __init__(self, beta=0.5):
        super(PixWiseBCELoss, self).__init__()
        self.criterion = tf.keras.losses.BinaryCrossentropy()
        self.beta = beta

    def call(self, y_true, y_pred):
        target_mask, target_label = y_true
        net_mask, net_label = y_pred

        pixel_loss = self.criterion(target_mask, net_mask)
        binary_loss = self.criterion(target_label, net_label)
        loss = pixel_loss * self.beta + binary_loss * (1 - self.beta)
        return loss
