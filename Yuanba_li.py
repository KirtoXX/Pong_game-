import tensorflow as tf
import tensorlayer as tl
import AI

class yunba:
    def __init__(self):
        self.network = None
        self.sess = tf.Session()
        self.weight_path = 'weight/20170407.npz'
        self.input = tf.placeholder(tf.float32, shape=[None,64,36,1])

    def build(self):
        self.netowk = AI.policy_network(self.input)
        load_params = tl.files.load_npz(path='weight/', name='20170407.npz')
        tl.files.assign_params(self.sess, load_params, self.network)
        init = tf.initialize_all_variables()
        self.sess.run(init)

    def close(self):
        self.sess.close()

    def action(self,observation):
        x = AI.preprocessing(observation)
        y = self.netowk.outputs

        result = self.sess.run(y,feed_dict={self.input:x})

        if result<0.5:
            result = 0
        else:
            result = 1

        return result



