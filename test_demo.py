import tensorflow as tf
import proc_input
import get_F1
#import matplotlib.pyplot as plt



batch_size = 2000
X, Y = proc_input.input_pipeline(['2nd-process/MN_data.csv'], batch_size)
# x_cv, y_cv = proc_input.input_pipeline(['cross_data.csv'], batch_size)
# x_test, y_test = proc_input.input_pipeline(['test_data.csv'], batch_size)

W = tf.Variable(tf.zeros([X.shape[1].value, 1], name='weight'))
b = tf.Variable([0.], name='bias')

alpha = 0.01
lam = 1.5

hyp = tf.sigmoid(tf.matmul(X, W) + b)
cost0 = Y * tf.log(hyp)
cost1 = (1 - Y) * tf.log(1 - hyp)
cost = (cost0 + cost1) / -batch_size
loss = (tf.reduce_sum(cost) + lam * tf.reduce_mean(tf.pow(W, 2))) / 2

loss1 = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=(tf.matmul(X, W) + b))) + lam * tf.reduce_mean(tf.pow(W, 2))) / 2

optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(loss1)
#基准值
base = 0.5
#转化成0or1
y_pred = tf.cast(tf.greater_equal(hyp, base), tf.float32)
accuracy , F1 = get_F1.tf_F1_score(Y, y_pred)

init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    sess.run(local_init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for eop in range(1000000):
        sess.run(optimizer)
        ls = sess.run(loss1)
        #if eop % 100 == 0:
        print(eop,"loss:", ls, " accuracy:" , sess.run(accuracy), " F1:" , sess.run(F1))
            #cost_accum.append(ls)

    coord.request_stop()
    coord.join(threads)
# plt.plot(range(len(cost_accum)),cost_accum,'r')
# plt.title('Logic Regression Cost Curve')
# plt.xlabel('epoch')
# plt.ylabel('cost')
# plt.show()