import tensorflow as tf

def cnn_model(features, labels, mode):
    # Input layer
    input_layer = tf.reshape(features, [-1, 160, 160, 1])
    
    # Convolutional layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[15, 15],
        padding="same",
        activation=tf.nn.relu
    )

    # Pooling layer 1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )

    # Convolutional layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=[15, 15],
        padding="same",
        activation=tf.nn.relu
    )

    # Pooling layer 2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )
    print(pool2)

    # Flatten layer
    flat = tf.reshape(pool2, [-1, 160*160])

    # Dense layer
    dense = tf.layers.dense(
        inputs=flat,
        units=256,
        activation=tf.nn.relu
    )

    # Logits layer
    logits = tf.layers.dense(
        inputs=dense,
        units=3
    )

    # Predictions and probabilities
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Training
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Evaluation
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"]),
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
