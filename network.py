import tensorflow as tf
import numpy as np

DEFAULT_PADDING = "SAME"

class Network(object):

    def __init__(self):
        pass
        # self.layers = dict(inputs)

    @staticmethod
    def set_weight_variable(shape, stddev=0.1):
        initial = tf.truncated_normal(shape, stddev)
        return tf.Variable(initial, name="weights")

    @staticmethod
    def set_bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name="bias")

    def conv(self, input, name, k_h, k_w,
             o_c, s_w, s_h, relu=True, padding=DEFAULT_PADDING):
        """
        k_w, k_h are kernal width and height
        o_c, i_c are in/output channels
        s_w, s_h are convolution stride width and height
        """
        i_c = input.get_shape().as_list()[-1]
        convolove = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.name_scope(name) as scope:
            weights = self.set_weight_variable(shape=[k_h, k_w, i_c, o_c])
            bias = self.set_bias_variable(shape=[o_c])
            # maybe add a group trainning
            # if group == 1:
            #     conv = convolove(input, weights)
            # else:
            #     split_input = tf.split(3, i_c, input)
            #     split_weights = tf.split(2, i_c, weights)
            #     split_conv = [convolove(i, k) for i, k in zip(split_input, split_weights)]
            #     print split_conv
            #     conv = tf.concat(3, split_conv)
            conv = convolove(input, weights)
            output = tf.nn.bias_add(conv, bias)
            if relu:
                return tf.nn.relu(output)
            else:
                return output

    def relu(self,input, name):
        return tf.nn.relu(input, name=name)

    def maxpool(self, input, name, k_h, k_w, s_h, s_w, padding=DEFAULT_PADDING):
        """
        k_h, k_w, s_h s_w are the same as conv definition
        """
        if padding not in ("SAME", "VALID"):
            raise TypeError("padding must be either SAME or VALID")
        with tf.variable_scope(name) as scope:
            return tf.nn.max_pool(input,
                                  kernel_size=[1, k_h, k_w,1],
                                  stride=[1, s_h, s_w, 1],
                                  padding=padding,
                                  name=name)

    def avgpool(self, input, name, k_h, k_w, s_h, s_w, padding=DEFAULT_PADDING):
        if padding not in ("SAME", "VALID"):
            raise TypeError("padding must be either SAME or VALID")
        with tf.variable_scope(name) as scope:
            return tf.nn.avg_pool(input,
                                  kernel_size=[i, k_h, k_w, 1],
                                  stride=[1, s_h, s_w, 1],
                                  padding=padding,
                                  name=name)

    def local_response_normalization(self, input, name, radius, alpah, beta, bias=1.0):
        with tf.variable_scope(name) as scope:
            return tf.nn.local_response_normalization(input,
                                                      depth_radius=radius,
                                                      alpah=alpah,
                                                      beta=beta,
                                                      bias=bias,
                                                      name=name)

    def fully_connected(self, input, name, num_out, relu=True):
        """
        num_out is the number of output channels after the fc layer
        """
        input_dims = input.get_shape().as_list()[1:]
        with tf.variable_scope(name) as scope:
            dims = reduce(lambda i, j: i*j, input_dims)
            input_flat = tf.reshape(input, shape=[-1, dims])
            weights = self.set_weight_variable(shape=[dims, num_out])
            bias = self.set_bias_variable(shape=[num_out])
            output = tf.nn.bias_add(tf.matmul(input_flat, weights), bias)
            if relu:
                output = tf.nn.relu(output)
            return output

    def dropout(self, input, name, prob):
        """
        prob is the probility of dropout
        """
        with tf.variable_scope(name) as scope:
            return tf.nn.dropout(input, prob, name=name)

    def softmax(self, input, name):
        with tf.variable_scope(name) as scope:
            return tf.nn.softmax(input, name)

    def cross_entropy(self, input, target, name):
        with tf.variable_scope(name) as scope:
            return -tf.reduce_sum(target*tf.log(input))

    def training(self, input, name, learning_rate):
        with tf.variable_scope(name) as scope:
            return tf.train.AdamOptimizer(learning_rate).minimize(input)

    def testing(self, input, name, target):
        with tf.variable_scope(name) as scope:
            correct_prediction = tf.equal(tf.argmax(input,1), tf.argmax(target, 1))
            return tf.reduce_mean(tf.cast(correct_prediction, "float"))

def main():
    net = Network()
    input_data, target = np.ones([2, 2352]), np.zeros([2, 10])
    target[0][7], target[1][5] = 1, 1
    print target
    x = tf.placeholder("float", shape=[None, 2352], name="x_input")
    x_images = tf.reshape(x, [-1,28,28,3])
    conv1 = net.conv(x_images, "conv1", 5, 5, 32, 1, 1)
    fc = net.fully_connected(conv1, "fc1", 10)
    drop = net.dropout(fc, "dropout", 0.5)
    soft = net.softmax(drop, "softmax")
    print soft, "soft"
    test = net.testing(soft, "testing", target)

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    y = soft.eval(feed_dict={x: input_data})
    z = test.eval(feed_dict={x: input_data})
    print y
    print z

if __name__ == "__main__":
    main()
