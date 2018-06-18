import numpy as np
import tensorflow as tf

class TextCNNRNN(object):
	def __init__(self, embedding_mat, sequence_length,num_classes, filter_sizes):

		self.embedding_mat = tf.constant(embedding_mat)#sho
		self.sequence_length = tf.constant(sequence_length)#sho
		self.num_classes = tf.constant(num_classes)#sho
		self.hidden_unit = 300#sho
		#self.non_static = tf.constant(False)#sho
		self.max_pool_size = 4#sho
		self.filter_sizes =filter_sizes#sho
		self.num_filters = 32#sho
		self.embedding_dim = 300#sho
		self.l2_reg_lambda = 0.0#sho
		self.input_x = tf.placeholder(tf.int32, [None,sequence_length], name='to_input_x')#done
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='to_input_y')#done
		self.dropout_keep_prob = tf.constant(0.5,name='to_dropout_keep_prob')#done
		self.batch_size = tf.constant(1,name='to_batch_size')#done
		self.pad = tf.constant(np.zeros([1, 1, 300, 1]), name='to_pad',dtype=tf.float32)#sho
		self.real_len =tf.constant([30])#sho

		l2_loss = tf.constant(0.0)

		with tf.device('/cpu:0'), tf.name_scope('embedding'):

			W = tf.constant(embedding_mat, name='W')

			self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
			emb = tf.expand_dims(self.embedded_chars, -1)

		pooled_concat = []
		reduced = np.int32(np.ceil((sequence_length) * 1.0 / 4))

		for i, filter_size in enumerate(filter_sizes):
			with tf.name_scope('conv-maxpool-%s' % filter_size):

				# Zero paddings so that the convolution output have dimension batch x sequence_length x emb_size x channel
				num_prio = (filter_size-1) // 2
				num_post = (filter_size-1) - num_prio
				pad_prio = tf.concat([self.pad] * num_prio,1)
				pad_post = tf.concat([self.pad] * num_post,1)
				emb_pad = tf.concat([pad_prio, emb, pad_post],1)

				filter_shape = [filter_size, 300, 1, 32]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
				b = tf.Variable(tf.constant(0.1, shape=[32]), name='b')
				conv = tf.nn.conv2d(emb_pad, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')

				h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(h, ksize=[1, self.max_pool_size, 1, 1], strides=[1, self.max_pool_size, 1, 1], padding='SAME', name='pool')
				pooled = tf.reshape(pooled, [-1, reduced, 32])
				pooled_concat.append(pooled)

		pooled_concat = tf.concat(pooled_concat,2)
		pooled_concat = tf.nn.dropout(pooled_concat, self.dropout_keep_prob)

		# lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_unit)

		#lstm_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_unit)
		lstm_cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_unit)

		#lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)
		lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)


		self._initial_state = lstm_cell.zero_state(self.batch_size, tf.float32)
		#inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, reduced, pooled_concat)]
		inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(pooled_concat,num_or_size_splits=int(reduced),axis=1)]
		#outputs, state = tf.nn.rnn(lstm_cell, inputs, initial_state=self._initial_state, sequence_length=self.real_len)
		outputs, state = tf.contrib.rnn.static_rnn(lstm_cell, inputs, initial_state=self._initial_state, sequence_length=self.real_len)

		# Collect the appropriate last words into variable output (dimension = batch x embedding_size)
		output = outputs[0]
		with tf.variable_scope('Output'):
			tf.get_variable_scope().reuse_variables()
			one = tf.ones([1, self.hidden_unit], tf.float32)
			for i in range(1,len(outputs)):
				ind = self.real_len < (i+1)
				ind = tf.to_float(ind)
				ind = tf.expand_dims(ind, -1)
				mat = tf.matmul(ind, one)
				output = tf.add(tf.multiply(output, mat),tf.multiply(outputs[i], 1.0 - mat))

		with tf.name_scope('output'):
			self.W = tf.Variable(tf.truncated_normal([self.hidden_unit, num_classes], stddev=0.1), name='W')
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			self.scores = tf.nn.xw_plus_b(output, self.W, b, name='scores')
			self.predictions = tf.argmax(self.scores, 1, name='to_predictions')

		with tf.name_scope('loss'):
			losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.input_y, logits = self.scores) #  only named arguments accepted
			self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

		with tf.name_scope('accuracy'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')

		with tf.name_scope('num_correct'):
			correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.num_correct = tf.reduce_sum(tf.cast(correct, 'float'))
