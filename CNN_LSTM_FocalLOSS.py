# Import necessary libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, MaxPooling1D, 
                                     LSTM, Dense, Dropout)
from tensorflow.keras.metrics import F1Score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler
import keras_tuner as kt
import datetime

# Define timesteps and number of features at each timestep
# Note that the X_train, X_validation, and X_test should be in the same shape as (-1,timesteps,features)
timesteps, features = 6, 24

class SparseCategoricalFocalLoss(Loss):
    """
    Numerically stable implementation of Focal Loss for sparse categorical labels.
    Supports per-class alpha for class imbalance.
    """
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        # Allow alpha to be scalar, list, tuple, or tensor
        # If a list or tuple or tensorf is passed, the elements represent the weight for each class
        if isinstance(alpha, (list, tuple)):
            # Normalize alpha
            total_alpha = sum(alpha)
            norm_alpha = [v / total_alpha for v in alpha]
            self.alpha = tf.convert_to_tensor(norm_alpha, dtype=tf.float32)
        elif isinstance(alpha, dict):
            # Normalize alpha
            sorted_keys = sorted(alpha.keys())
            values_ = [alpha[k] for k in sorted_keys]
            total_alpha = sum(values_)
            norm_alpha = [v / total_alpha for v in values_]
            self.alpha =tf.convert_to_tensor(norm_alpha, dtype=tf.float32)
            
        else:
            self.alpha = alpha
        self.from_logits = from_logits
      
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.int32)
        y_true = tf.squeeze(y_true, axis=-1) if tf.rank(y_true) > 1 else y_true

        if self.from_logits:
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            probs = tf.nn.softmax(y_pred, axis=-1)
        else:
            ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            probs = y_pred

        batch_indices = tf.range(tf.shape(y_true)[0], dtype=tf.int32)
        indices = tf.stack([batch_indices, y_true], axis=1)
        p_t = tf.gather_nd(probs, indices)

        modulating_factor = tf.pow(1.0 - p_t, self.gamma)

        # Handle per-class alpha
        if isinstance(self.alpha, tf.Tensor) and self.alpha.shape.rank == 1:
            alpha_factor = tf.gather(self.alpha, y_true)  # Gather alpha for each true class
        else:
            alpha_factor = self.alpha if self.alpha is not None else 1.0

        focal_loss = alpha_factor * modulating_factor * ce
        return tf.reduce_mean(focal_loss)

    def get_config(self):
        base_config = super().get_config()
        # Serialize alpha appropriately
        alpha_config = self.alpha.numpy().tolist() if isinstance(self.alpha, tf.Tensor) else self.alpha
        return {**base_config, "gamma": self.gamma, "alpha": alpha_config, "from_logits": self.from_logits}

# Custom Learning Rate Scheduler
def lr_step_decay(epoch,lr):
    """
    Learning rate scheduler function.
    """
    if epoch < 10:
        return 0.01
    elif epoch < 20:
        return 0.005
    elif epoch < 50:
        return 0.001
    elif epoch < 80:
        return 0.0005
    else:
        return 0.0001

from tensorflow.keras.metrics import F1Score

class SparseF1Score(F1Score):
    def __init__(self, num_classes=7, average='weighted', name='f1_score', **kwargs):
        super().__init__(average=average, name=name, **kwargs)
        self.num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Robust handling of y_true: Cast to int32, squeeze extra dims if present
        y_true = tf.cast(y_true, tf.int32)
        if tf.rank(y_true) > 1:
            y_true = tf.squeeze(y_true, axis=-1)
        # One-hot encode
        y_true_one_hot = tf.one_hot(y_true, depth=self.num_classes)
        # Convert logits to probabilities
        y_pred_probs = tf.nn.softmax(y_pred, axis=-1)
        # Call the superclass update_state
        super().update_state(y_true_one_hot, y_pred_probs, sample_weight=sample_weight)


def build_model(hp):
    inputs = Input(shape=(timesteps, features))

    # Define L2 Regularization parameter
    hp_l2 = hp.Float('l2_reg', min_value=1e-5, max_value=1e-2, sampling='log')
    # Convolutional Feature Extractor Block
    hp_filters = hp.Int('conv_filters', min_value=32, max_value=128, step=32)
    hp_kernel_size = hp.Choice('conv_kernel_size', values=[2, 3])
    x = Conv1D(filters=hp_filters, kernel_size=hp_kernel_size, activation='relu', padding='same', kernel_regularizer=L2(hp_l2))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Recurrent Block 
    hp_lstm1_units = hp.Int('lstm_1_units', min_value=32, max_value=128, step=32)
    x = LSTM(units=hp_lstm1_units, return_sequences=True, kernel_regularizer=L2(hp_l2), recurrent_regularizer=L2(hp_l2))(x)
    hp_dropout_1 = hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)
    x = Dropout(hp_dropout_1)(x)
   
    hp_lstm2_units = hp.Int('lstm_2_units', min_value=32, max_value=128, step=32)
    x = LSTM(units=hp_lstm2_units, return_sequences=False, kernel_regularizer=L2(hp_l2), recurrent_regularizer=L2(hp_l2))(x)
    hp_dropout_2 = hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)
    x = Dropout(hp_dropout_2)(x)

    # Classifier Head
    hp_dense_units = hp.Int('dense_units', min_value=32, max_value=128, step=32)
    x = Dense(units=hp_dense_units, activation='relu', kernel_regularizer=L2(hp_l2))(x)

    outputs = Dense(7, kernel_regularizer=L2(hp_l2))(x)
    model = Model(inputs=inputs, outputs=outputs)

    # Focal Loss definition which uses per-class alpha from the dictionary.
    hp_gamma = hp.Float('focal_loss_gamma', min_value=1.0, max_value=5.0, step=0.5)
    loss_fn = SparseCategoricalFocalLoss(
        gamma=hp_gamma,
        alpha=class_weights_dict,  # Pass the weights dictionary directly (handled in __init__)
        from_logits=True  # Explicitly tell the loss to expect logits.
    )

    initial_learning_rate = 0.01
   
    model.compile(
        optimizer=Adam(learning_rate=initial_learning_rate),
        loss=loss_fn,  # Use our custom focal loss class instance
        metrics=[SparseF1Score(num_classes=7, average='weighted', name='f1_score')]
    )

    return model

# Hyperparameter Tuning using Bayesian Optimization
tuner = kt.BayesianOptimization(
    build_model,
    objective=kt.Objective("val_f1_score", direction="max"),
    max_trials=100,
    executions_per_trial=3,
    directory='lulc_tuning_advanced',
    project_name='cnn_lstm_focal_loss'
)

# Create a unique directory for TensorBoard logs for each run
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Define all callbacks
callbacks = [
    EarlyStopping(monitor='val_f1_score', mode='max', patience=10, restore_best_weights=True),
    TensorBoard(log_dir=log_dir, histogram_freq=1),
    LearningRateScheduler(lr_step_decay, verbose=1) # Add our LR scheduler
]

# Start the search
tuner.search(X_train_reshaped, Y_train,
             epochs=150, # A high number of epochs; scheduler and early stopping will manage it
             validation_data=(X_val_reshaped, Y_val),
             callbacks=callbacks,
             batch_size=256)

# --- 6. Retrieve and Evaluate the Best Model ---
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

test_results = best_model.evaluate(X_test_reshaped, Y_test)
print(f"Test Results - Loss: {test_results[0]} - F1-Score: {test_results[1]}")
