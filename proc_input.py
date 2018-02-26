import tensorflow as tf

# 数据读取
# 数据tensor化

def read_my_file_format(filename_queue):
    reader = tf.TextLineReader()
    key, record_string = reader.read(filename_queue)
    defaults = [[0.], [1.], [10.], [0.]]
    col1, col2, col3, col4 = tf.decode_csv(record_string, record_defaults=defaults)
    features = tf.stack([col1, col2, col3])
    # features = tf.stack(set_pdynome_degree(3, [col1, col2, col3]))
    features = tf.reshape(features, [-1, 1])
    lable = tf.stack([col4])
    return features, lable


def input_pipeline(filenames, batch_size, num_epochs = None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    example, lable = read_my_file_format(filename_queue)
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, lable_batch = tf.train.shuffle_batch(
        [example, lable], batch_size = batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )
    return example_batch, lable_batch



# def test_input_main():
#     x, y = input_pipeline(['test_data.csv'], 10000, 1)
#     init_op = tf.global_variables_initializer()
#     local_init_op = tf.local_variables_initializer()
#
#     sess = tf.Session()
#     sess.run(init_op)
#     sess.run(local_init_op)
#
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess = sess, coord=coord)
#
#     try:
#         while not coord.should_stop():
#             #sess.run([x, y])
#             print(sess.run(tf.shape(x)))
#     except tf.errors.OutOfRangeError:
#         print('Done training -- epoch limit reached')
#     finally:
#         coord.request_stop()
#
#     coord.join(threads)
#     sess.close()
#
# test_input_main()

