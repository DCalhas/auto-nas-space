import tensorflow as tf

import gc

from utils import print_utils

def train_step(model, x, y, optimizer, loss_fn, return_logits=False):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    @tf.function
    def apply_gradient(model, optimizer, loss_fn, x, y, return_logits=False):
        with tf.GradientTape() as tape:
            logits = model(x)
            loss = loss_fn(y, logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if(return_logits):
            return (tf.reduce_mean(loss), logits)
        return tf.reduce_mean(loss)

    return apply_gradient(model, optimizer, loss_fn, x, y, return_logits=return_logits)

def evaluate(eval_set, model, loss_fn):
    loss = 0.0
    acc = 0.0
    n_batches = 0
    instances = 0
    for batch_x, batch_y in eval_set.repeat(1):
        batch_pred = model(batch_x)
        loss += tf.reduce_mean(loss_fn(batch_y, batch_pred)).numpy()
        acc += int(tf.argmax(batch_pred, axis=-1) == tf.argmax(batch_y, axis=-1))
        n_batches += 1
    
    loss = loss/n_batches
    acc = acc/n_batches

    loss_std = 0.0
    acc_std = 0.0
    for batch_x, batch_y in eval_set.repeat(1):
        batch_pred = model(batch_x)
        loss_std += (loss-tf.reduce_mean(loss_fn(batch_y, batch_pred)).numpy())**2
        acc_std += (acc-int(tf.argmax(batch_pred, axis=-1) == tf.argmax(batch_y, axis=-1)))**2

    loss_std=(loss_std/n_batches)**(1/2)
    acc_std=(acc_std/n_batches)**(1/2)
    
    return loss, loss_std, acc, acc_std


def train(train_set, model, opt, loss_fn, epochs=10, val_set=None, file_output=None, verbose=False):
    val_history = []
    train_loss = []

    for epoch in range(epochs):

        loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y in train_set.repeat(1):
            loss += train_step(model, batch_x, batch_y, opt, loss_fn).numpy()
            n_batches += 1
            gc.collect()

        if(val_set is not None):
            val_history.append(evaluate(val_set, model, loss_fn))
        train_loss.append(loss/n_batches)

        print_utils.print_message("Epoch " + str(epoch+1) + " with loss: " + str(train_loss[-1]), file_output=file_output, verbose=verbose)

    return train_loss, val_history