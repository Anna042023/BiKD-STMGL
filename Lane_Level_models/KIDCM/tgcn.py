# tgcn.py — TF2.20 兼容（使用 v1 行为），保留 T-GCN 单元原逻辑
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from utils import calculate_laplacian
# ★ 关键修正：从 TF v1 兼容命名空间导入 RNNCell
from tensorflow.compat.v1.nn.rnn_cell import RNNCell


class tgcnCell(RNNCell):
    """Temporal Graph Convolutional Network Cell (GCN + GRU gates)."""

    def __init__(self, num_units, adj, num_nodes, input_size=None,
                 act=tf.nn.tanh, reuse=None):
        # 兼容 TF1 风格
        super(tgcnCell, self).__init__(_reuse=reuse)
        self._act = act
        self._nodes = num_nodes
        self._units = num_units
        # 归一化 (I + A) -> 稀疏张量，供图传播用
        self._adj = [calculate_laplacian(adj)]

    @property
    def state_size(self):
        # 按 TF1 RNNCell 规范：整个状态大小（展平后）
        return self._nodes * self._units

    @property
    def output_size(self):
        # 每个时间步的输出特征维度（每节点共享）
        return self._units

    # TF1 RNNCell 接口：call(self, inputs, state) -> (output, new_state)
    def call(self, inputs, state):
        with tf.variable_scope("tgcn"):
            with tf.variable_scope("gates"):
                # 门控（r, u）
                value = tf.nn.sigmoid(self._gc(inputs, state, 2 * self._units, bias=1.0))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
            with tf.variable_scope("candidate"):
                r_state = r * state
                c = self._act(self._gc(inputs, r_state, self._units))
            new_h = u * state + (1 - u) * c
        return new_h, new_h

    # 兼容旧代码里直接 __call__ 的用法
    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            return self.call(inputs, state)

    def _gc(self, inputs, state, output_size, bias=0.0):
        """
        图卷积作用在 concat([inputs, state], dim=-1) 上。
        inputs: (B, N)
        state:  (B, N*U)
        return: (B, N*output_size)
        """
        # inputs -> (B, N, 1)
        inputs = tf.expand_dims(inputs, 2)
        # state -> (B, N, U)
        state = tf.reshape(state, (-1, self._nodes, self._units))
        # (B, N, 1+U)
        x_s = tf.concat([inputs, state], axis=2)
        input_size = x_s.get_shape()[2].value  # 1 + U

        # (N, input_size, B) -> (N, B*input_size)
        x0 = tf.transpose(x_s, perm=[1, 2, 0])
        x0 = tf.reshape(x0, shape=[self._nodes, -1])

        # 稀疏图传播 (I + A) 归一化后的稀疏邻接
        for m in self._adj:
            x1 = tf.sparse_tensor_dense_matmul(m, x0)  # (N, B*input_size)

        # (N, C, B) -> (B, N, C) -> (B*N, C)
        x = tf.reshape(x1, shape=[self._nodes, input_size, -1])
        x = tf.transpose(x, perm=[2, 0, 1])
        x = tf.reshape(x, shape=[-1, input_size])

        # 线性变换
        weights = tf.get_variable(
            'weights', [input_size, output_size],
            initializer=tf.glorot_uniform_initializer())
        x = tf.matmul(x, weights)
        biases = tf.get_variable(
            "biases", [output_size],
            initializer=tf.constant_initializer(bias, dtype=tf.float32))
        x = tf.nn.bias_add(x, biases)

        # (B, N, output_size) -> (B, N*output_size)
        x = tf.reshape(x, shape=[-1, self._nodes, output_size])
        x = tf.reshape(x, shape=[-1, self._nodes * output_size])
        return x
