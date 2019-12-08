import numpy as np
import tensorflow as tf

a = tf.constant(np.random.rand(2, 3, 4))
sess = tf.Session()

# ms = sess.run(tf.nn.top_k(a, k=2))
ms = tf.nn.top_k(a, k=2)
# input = sess.run(a)
# print(ms)
def k_max(ms, input):
    sh = ms.values.shape
    ind_sort = np.sort(tf.constant(ms.indices).eval())
    zs = np.zeros((sh))
    for b in range(sh[0]):
        for r in range(sh[1]):
            for c in range(sh[2]):
                zs[b][r][c] = input[b][r][ind_sort[b][r][c]]
    return zs
input = a.eval()
zss = k_max(ms, input)
print(zss)
tf.unsorted_segment_max()
