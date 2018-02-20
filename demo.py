import tensorflow as tf
import proc_input
#import matplotlib.pyplot as plt
import numpy as np


batch_size = 1000
X, y = proc_input.input_pipeline(['tarin_data.csv'], batch_size, 1)
x_cross, y_cross = proc_input.input_pipeline(['cross_data.csv'], batch_size)
x_test, y_test = proc_input.input_pipeline(['test_data.csv'], batch_size)


W = tf.Variable(tf.zeros([2,1], name='weight'))
b = tf.Variable([0.], name='bias')

hyp = tf.sigmoid(tf.matmul(X, W) + b)
cost0 = y * tf.log(hyp)
cost1 = (1 - y) * tf.log(hyp)
cost = (cost0 + cost1) / -batch_size
loss = tf.reduce_sum(cost)

loss1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=(tf.matmul(X, W) + b)) / batch_size)
alpha = 0.000001

optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(loss1)
#基准值
base = 0.6
#转化成0or1
y_pred = tf.cast(tf.greater_equal(hyp, base), tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_pred), tf.float32))
#F1-score
################

cost_accum = []
cost_prev = 100

init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    sess.run(local_init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for eop in range(1000000):
        sess.run(optimizer)
        if eop % 100 == 0:
            ls = sess.run(loss1)
            print(eop, sess.run(W).flatten(), sess.run(b).flatten(), ls, sess.run(accuracy))
            cost_accum.append(ls)

    coord.request_stop()
    coord.join(threads)
# plt.plot(range(len(cost_accum)),cost_accum,'r')
# plt.title('Logic Regression Cost Curve')
# plt.xlabel('epoch')
# plt.ylabel('cost')
# plt.show()