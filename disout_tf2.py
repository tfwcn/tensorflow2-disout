import tensorflow as tf


class Disout(tf.keras.layers.Layer):
    '''
    disout
    论文：https://arxiv.org/abs/2002.11022
    '''

    def __init__(self, dist_prob, block_size=5, alpha=1, **kwargs):
        super(Disout, self).__init__(**kwargs)
        self.dist_prob = dist_prob
        self.weight_behind=None

        self.alpha = alpha
        self.block_size = block_size

    def build(self, input_shape):
        pass

    @tf.function
    def call(self, x):
        '''x：(batch_size,h,w,c)'''
        if not self.trainable:
            return x
        else:
            if tf.math.equal(tf.rank(x),4):
                x_shape = tf.shape(x)
                x_size = x_shape[1:3]
                x_size_f = tf.cast(x_size, tf.float32)
                # 计算block_size
                x_block_size_f = tf.constant((self.block_size, self.block_size), tf.float32)
                # x_block_size_f = x_size_f * self.block_size
                # x_block_size_f = tf.math.maximum(x_block_size_f, 1)
                x_block_size = tf.cast(x_block_size_f, tf.int32)
                # 根据dist_prob，计算block_num
                x_block_num = (x_size_f[0] * x_size_f[1]) * self.dist_prob / (x_block_size_f[0] * x_block_size_f[1])
                # 计算block在中心区域出现的概率
                x_block_rate = x_block_num / ((x_size_f[0] - x_block_size_f[0] + 1) * (x_size_f[1] - x_block_size_f[1] + 1))
                # tf.print('x_block_rate:', x_block_rate)
                # 根据概率生成block区域
                x_block_center = tf.random.uniform((x_shape[0], x_size[0] - x_block_size[0] + 1, x_size[1] - x_block_size[1] + 1, x_shape[3]), dtype=tf.float32)
                x_block_padding_t = x_block_size[0] // 2
                x_block_padding_b = x_size_f[0] - tf.cast(x_block_padding_t, tf.float32) - (x_size_f[0] - x_block_size_f[0] + 1.0)
                x_block_padding_b = tf.cast(x_block_padding_b, tf.int32)
                x_block_padding_l = x_block_size[1] // 2
                x_block_padding_r = x_size_f[1] - tf.cast(x_block_padding_l, tf.float32) - (x_size_f[1] - x_block_size_f[1] + 1.0)
                x_block_padding_r = tf.cast(x_block_padding_r, tf.int32)
                x_block_padding = tf.pad(x_block_center,[[0, 0],[x_block_padding_t, x_block_padding_b],[x_block_padding_l, x_block_padding_r],[0, 0]])
                x_block = tf.cast(x_block_padding<x_block_rate, tf.float32)
                x_block = tf.nn.max_pool2d(x_block, ksize=[self.block_size, self.block_size], strides=[1, 1], padding='SAME')
                # block百分比
                # x_block_percent_ones = tf.reduce_sum(x_block) / tf.reduce_prod(tf.cast(tf.shape(x_block), tf.float32))
                # tf.print('x_block_percent_ones:', x_block_percent_ones, tf.shape(x_block))
                # 特征叠加
                x_abs = tf.abs(x)
                x_sum = tf.math.reduce_sum(x_abs, axis=-1, keepdims=True)
                x_max = tf.math.reduce_max(x_sum, axis=(1, 2), keepdims=True)
                x_max_c = tf.math.reduce_max(x_abs, axis=(1, 2), keepdims=True)
                x_sum_c = tf.math.reduce_sum(x_max_c, axis=-1, keepdims=True)
                x_v = x_sum / x_sum_c
                # tf.print('x_v:', tf.shape(x_v), tf.math.reduce_min(x_v), tf.math.reduce_max(x_v))
                # 特征方差
                # x_variance = tf.math.reduce_variance(x_sum, axis=(1, 2), keepdims=True)
                # tf.print('x_variance:', tf.shape(x_variance), tf.math.reduce_min(x_variance), tf.math.reduce_max(x_variance))
                # 叠加扰动
                x_max = tf.reduce_max(x, axis=(1,2), keepdims=True)
                x_min = tf.reduce_min(x, axis=(1,2), keepdims=True)
                x_block_random = tf.random.uniform(x_shape, dtype=x.dtype) * (x_max - x_min) + x_min
                x_block_random = x_block_random * (self.alpha * x_v + 0.3) + x * (1.0 - self.alpha * x_v - 0.3)
                x = x * (1-x_block) + x_block_random * x_block

                # if not (self.weight_behind is None) and not(len(self.weight_behind)==0):
                #     wtsize=tf.shape(self.weight_behind[0])[0]
                #     weight_max=tf.math.reduce_max(self.weight_behind[0], axis=-2, keepdims=True)
                #     sig = tf.ones(tf.shape(weight_max),dtype=weight_max.dtype)
                #     sig_mask = tf.cast(tf.random.uniform(tf.shape(weight_max),dtype=sig.dtype)<0.5,dtype=tf.float32)
                #     sig = sig * (1 - sig_mask) - sig_mask
                #     weight_max = weight_max * sig 
                #     weight_mean = tf.math.reduce_mean(weight_max, axis=(0,1), keepdims=True)
                #     if wtsize==1:
                #         weight_mean=0.1*weight_mean
                #     #print(weight_mean)
                # mean=tf.math.reduce_mean(x)
                # var=tf.math.reduce_variance(x)

                # if not (self.weight_behind is None) and not(len(self.weight_behind)==0):
                #     dist=self.alpha*weight_mean*(var**0.5)*tf.random.normal(tf.shape(x), dtype=x.dtype)
                # else:
                #     dist=self.alpha*0.01*(var**0.5)*tf.random.normal(tf.shape(x), dtype=x.dtype)

                # x=x*x_block
                # dist=dist*(1-x_block)
                # x=x+dist
                # x=x/x_block_percent_ones
                return x
            else:
                return x

    def compute_output_shape(self, input_shape):
        '''计算输出shape'''
        return input_shape

class Disout1D(tf.keras.layers.Layer):
    '''
    disout
    论文：https://arxiv.org/abs/2002.11022
    '''

    def __init__(self, dist_prob, block_size=5, alpha=0.5, **kwargs):
        super(Disout1D, self).__init__(**kwargs)
        self.dist_prob = dist_prob

        self.alpha = alpha
        self.block_size = block_size

    def build(self, input_shape):
        pass

    @tf.function
    def call(self, x):
        '''x：(batch_size,h,w,c)'''
        if not self.trainable:
            return x
        else:
            if tf.math.equal(tf.rank(x),2):
                x_shape = tf.shape(x)
                x_size = x_shape[1]
                x_size_f = tf.cast(x_size, tf.float32)
                # 计算block_size
                x_block_size_f = tf.constant(self.block_size, tf.float32)
                # x_block_size_f = x_size_f * self.block_size
                # x_block_size_f = tf.math.maximum(x_block_size_f, 1)
                x_block_size = tf.cast(x_block_size_f, tf.int32)
                # 根据dist_prob，计算block_num
                x_block_num = (x_size_f) * self.dist_prob / (x_block_size_f)
                # 计算block在中心区域出现的概率
                x_block_rate = x_block_num / ((x_size_f - x_block_size_f + 1))
                # 根据概率生成block区域
                x_block_center = tf.random.uniform((x_shape[0], x_size - x_block_size + 1), dtype=tf.float32)
                x_block_padding_t = x_block_size // 2
                x_block_padding_b = x_size_f - tf.cast(x_block_padding_t, tf.float32) - (x_size_f - x_block_size_f + 1.0)
                x_block_padding_b = tf.cast(x_block_padding_b, tf.int32)
                x_block_padding = tf.pad(x_block_center,[[0, 0],[x_block_padding_t, x_block_padding_b]])
                x_block = tf.cast(x_block_padding<x_block_rate, tf.float32)
                x_block = tf.expand_dims(x_block, axis=-1)
                x_block = tf.nn.max_pool1d(x_block, ksize=[self.block_size], strides=[1], padding='SAME')
                x_block = tf.reshape(x_block, x_shape)
                # 叠加扰动
                x_max = tf.reduce_max(x, axis=1, keepdims=True)
                x_min = tf.reduce_min(x, axis=1, keepdims=True)
                x_block_random = tf.random.uniform(x_shape, dtype=x.dtype) * (x_max - x_min) + x_min
                x_block_random = x_block_random * (1.0 - self.alpha) + x * self.alpha
                x = x * (1-x_block) + x_block_random * x_block
                return x
            else:
                return x

    def compute_output_shape(self, input_shape):
        '''计算输出shape'''
        return input_shape
