import tensorflow as tf
from disout_tf2 import Disout, Disout1D


class CustomLayer(tf.keras.layers.Layer):
    '''自定义层'''

    def __init__(self, units, activation, **args):
        super(CustomLayer, self).__init__(**args)
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        '''初始化网络'''
        num_inputs = input_shape[-1]
        self.kernel = self.add_weight('kernel',
                                      shape=[num_inputs,
                                             self.units],
                                      initializer=tf.keras.initializers.he_normal()
                                      )
        self.bias = self.add_weight("bias",
                                    shape=[self.units],
                                    initializer=tf.keras.initializers.Zeros
                                    )
        self.activation_layer = tf.keras.layers.Activation(self.activation)

    def call(self, x):
        '''运算部分'''
        print('call动态图会不断打印该信息，静态图只打印数次。', type(x), x.shape)
        output = tf.matmul(x, self.kernel) + self.bias
        output = self.activation_layer(output)
        return output

    def compute_output_shape(self, input_shape):
        '''计算输出shape'''
        return (input_shape[0], self.output_dim)


class CustomModel(tf.keras.Model):
    '''自定义模型'''

    def __init__(self):
        '''初始化模型层'''
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=5, activation='relu')
        self.disout1 = Disout(0.05, block_size=3)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, activation='relu')
        self.disout2 = Disout(0.05, block_size=2)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=2)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = CustomLayer(units=128, activation='relu')
        # self.dropout = tf.keras.layers.Dropout(rate=0.7)
        self.disout3 = Disout1D(0.3, block_size=1)
        self.fc2 = CustomLayer(units=10, activation='softmax')

    def call(self, x):
        '''运算部分'''
        x = self.conv1(x)
        # self.disout1.weight_behind = self.conv1.weights
        x = self.disout1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        # self.disout2.weight_behind = self.conv2.weights
        x = self.disout2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        # x = self.dropout(x)
        x = self.disout3(x)
        return self.fc2(x)

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        if 'val_accuracy' in logs:
            for layer in self.model.layers:
                if isinstance(layer, Disout):
                    layer.alpha = float(logs['val_accuracy'])
            # print('on_epoch_end update layer.alpha:', logs['val_accuracy'])
        pass

def main():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = tf.expand_dims(
        x_train, axis=-1), tf.expand_dims(x_test, axis=-1)
    y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)
    print('x_train, y_train', x_train.shape, y_train.shape)
    print('x_test, y_test', x_test.shape, y_test.shape)

    # 卷积实现图片分类
    model = CustomModel()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[CustomCallback()])


if __name__ == '__main__':
    # 设置GPU显存自适应
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
    print(gpus, cpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    # tf.config.experimental.set_visible_devices(cpus[0], 'CPU')
    main()
