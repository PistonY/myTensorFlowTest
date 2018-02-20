import tensorflow as tf

#输入输出均为tensor
#输入的tensor为0or1的M*1矩阵
def tf_F1_score(actuals, predictions):
    actuals = tf.reshape(actuals, [-1, 1])
    predictions = tf.reshape(predictions, [-1, 1])

    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)

    #true-positive
    tp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            dtype=tf.float32
        )
    )
    #true-Negative
    tn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            dtype=tf.float32
        )
    )
    #false-positive
    fp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            dtype=tf.float32
        )
    )
    #false_Neg
    fn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            dtype=tf.float32
        )
    )

    accuracy = (tp_op + tn_op) / (tp_op + tn_op + fp_op + fn_op)
    prediction = tp_op / (tp_op + fp_op)
    recall = tp_op / (tp_op + fn_op)
    f1_score = (2 * (prediction * recall)) / (prediction + recall)

    return accuracy, [tp_op, tn_op, fp_op, fn_op, f1_score]