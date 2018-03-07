import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sets
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export

# def metric_variable(shape, dtype, validate_shape=True, name=None):
#   """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES`) collections."""

#   return variable_scope.variable(
#       lambda: array_ops.zeros(shape, dtype),
#       trainable=False,
#       collections=[
#           ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.METRIC_VARIABLES
#       ],
#       validate_shape=validate_shape,
#       name=name)

with tf.Graph().as_default():
	a = tf.placeholder(tf.float32, [])
	loss = a * 2

	streaming_loss, update_op = tf.metrics.mean(loss)


	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	loss_val, _ = sess.run([loss, update_op], feed_dict={a:2})
	print(loss_val, sess.run(streaming_loss))

	loss_val, _ = sess.run([loss, update_op], feed_dict={a:1})
	print(loss_val, sess.run(streaming_loss))

	loss_val, _ = sess.run([loss, update_op], feed_dict={a:3})
	print(loss_val, sess.run(streaming_loss))

	loss_val, _ = sess.run([loss, update_op], feed_dict={a:5})
	print(loss_val, sess.run(streaming_loss))