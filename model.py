import tensorflow as tf
import pickle

class Model(object):
    
	def __init__(self, user_count, item_count):

		self.user = tf.compat.v1.placeholder(tf.compat.v1.int32, [None,]) # [B]
		self.item = tf.compat.v1.placeholder(tf.compat.v1.int32, [None,]) # [B]
		self.label = tf.compat.v1.placeholder(tf.compat.v1.float32, [None,]) # [B]

		self.u_read = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, None]) # [B, R]
		self.u_read_l = tf.compat.v1.placeholder(tf.compat.v1.int32, [None,]) # [B]
		self.u_friend = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, None]) # [B, F]
		self.u_friend_l = tf.compat.v1.placeholder(tf.compat.v1.int32, [None,]) # [B]
		self.uf_read = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, None, None]) # [B, F, R]
		self.uf_read_l = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, None]) # [B, F]

		self.i_read = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, None]) # [B, R]
		self.i_read_l = tf.compat.v1.placeholder(tf.compat.v1.int32, [None,]) # [B]
		self.i_friend = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, None]) # [B, R]
		self.i_friend_l = tf.compat.v1.placeholder(tf.compat.v1.int32, [None,]) # [B]
		self.if_read = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, None, None]) # [B, F, R]
		self.if_read_l = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, None]) # [B, F]
		self.i_link = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, None, 1]) # [B, F, 1]

		self.learning_rate = tf.compat.v1.placeholder(tf.compat.v1.float32)
		self.training = tf.compat.v1.placeholder(tf.compat.v1.int32)
		self.keep_prob = tf.compat.v1.placeholder(tf.compat.v1.float32)
		self.lambda1 = tf.compat.v1.placeholder(tf.compat.v1.float32)
		self.lambda2 = tf.compat.v1.placeholder(tf.compat.v1.float32)

		#--------------embedding layer-------------------
        
		hidden_units_u = 10 # user embedding size
		hidden_units_i = 10 # item embedding size

		user_emb_w = tf.compat.v1.get_variable("norm_user_emb_w", [user_count+1, hidden_units_u], initializer = None)
		item_emb_w = tf.compat.v1.get_variable("norm_item_emb_w", [item_count+1, hidden_units_i], initializer = None)
		item_b = tf.compat.v1.get_variable("norm_item_b", [item_count+1],
                             initializer=tf.compat.v1.constant_initializer(0.0))

		# embedding for user and item
		uid_emb = tf.compat.v1.nn.embedding_lookup(user_emb_w, self.user)
		iid_emb = tf.compat.v1.nn.embedding_lookup(item_emb_w, self.item)
		i_b = tf.compat.v1.gather(item_b, self.item)

		# embedding for user's clicked items
		ur_emb = tf.compat.v1.nn.embedding_lookup(item_emb_w, self.u_read) # [B, R, H]
		key_masks = tf.compat.v1.sequence_mask(self.u_read_l, tf.compat.v1.shape(ur_emb)[1])   # [B, R]
		key_masks = tf.compat.v1.expand_dims(key_masks, axis = 2) # [B, R, 1]
		key_masks = tf.compat.v1.tile(key_masks, [1, 1, tf.compat.v1.shape(ur_emb)[2]]) # [B, R, H]
		key_masks = tf.compat.v1.reshape(key_masks, [-1, tf.compat.v1.shape(ur_emb)[1], tf.compat.v1.shape(ur_emb)[2]]) # [B, R, H]
		paddings = tf.compat.v1.zeros_like(ur_emb) # [B, R, H]
		ur_emb = tf.compat.v1.where(key_masks, ur_emb, paddings)  # [B, R, H]

		# embedding for item's clicking users
		ir_emb = tf.compat.v1.nn.embedding_lookup(user_emb_w, self.i_read) # [B, R, H]
		key_masks = tf.compat.v1.sequence_mask(self.i_read_l, tf.compat.v1.shape(ir_emb)[1])   # [B, R]
		key_masks = tf.compat.v1.expand_dims(key_masks, axis = 2) # [B, R, 1]
		key_masks = tf.compat.v1.tile(key_masks, [1, 1, tf.compat.v1.shape(ir_emb)[2]]) # [B, R, H]
		key_masks = tf.compat.v1.reshape(key_masks, [-1, tf.compat.v1.shape(ir_emb)[1], tf.compat.v1.shape(ir_emb)[2]]) # [B, R, H]
		paddings = tf.compat.v1.zeros_like(ir_emb) # [B, R, H]
		ir_emb = tf.compat.v1.where(key_masks, ir_emb, paddings)  # [B, R, H]
        
		# embedding for user's friends
		fuid_emb = tf.compat.v1.nn.embedding_lookup(user_emb_w, self.u_friend)
		key_masks = tf.compat.v1.sequence_mask(self.u_friend_l, tf.compat.v1.shape(fuid_emb)[1])   # [B, F]
		key_masks = tf.compat.v1.expand_dims(key_masks, axis = 2) # [B, F, 1]
		key_masks = tf.compat.v1.tile(key_masks, [1, 1, tf.compat.v1.shape(fuid_emb)[2]]) # [B, F, H]
		paddings = tf.compat.v1.zeros_like(fuid_emb) # [B, F, H]
		fuid_emb = tf.compat.v1.where(key_masks, fuid_emb, paddings)  # [B, F, H]

		# embedding for item's related items
		fiid_emb = tf.compat.v1.nn.embedding_lookup(item_emb_w, self.i_friend)
		key_masks = tf.compat.v1.sequence_mask(self.i_friend_l, tf.compat.v1.shape(fiid_emb)[1])   # [B, F]
		key_masks = tf.compat.v1.expand_dims(key_masks, axis = 2) # [B, F, 1]
		key_masks = tf.compat.v1.tile(key_masks, [1, 1, tf.compat.v1.shape(fiid_emb)[2]]) # [B, F, H]
		paddings = tf.compat.v1.zeros_like(fiid_emb) # [B, F, H]
		fiid_emb = tf.compat.v1.where(key_masks, fiid_emb, paddings)  # [B, F, H]

		# embedding for user's friends' clicked items
		ufr_emb = tf.compat.v1.nn.embedding_lookup(item_emb_w, self.uf_read)
		key_masks = tf.compat.v1.sequence_mask(self.uf_read_l, tf.compat.v1.shape(ufr_emb)[2])   # [B, F, R]
		key_masks = tf.compat.v1.expand_dims(key_masks, axis = 3) # [B, F, R, 1]
		key_masks = tf.compat.v1.tile(key_masks, [1, 1, 1, tf.compat.v1.shape(ufr_emb)[3]]) # [B, F, R, H]
		paddings = tf.compat.v1.zeros_like(ufr_emb) # [B, F, R, H]
		ufr_emb = tf.compat.v1.where(key_masks, ufr_emb, paddings)  # [B, F, R, H]

		# embedding for item's related items' clicking users
		ifr_emb = tf.compat.v1.nn.embedding_lookup(user_emb_w, self.if_read) # [B, F, R, H]
		key_masks = tf.compat.v1.sequence_mask(self.if_read_l, tf.compat.v1.shape(ifr_emb)[2])   # [B, F, R]
		key_masks = tf.compat.v1.expand_dims(key_masks, axis = 3) # [B, F, R, 1]
		key_masks = tf.compat.v1.tile(key_masks, [1, 1, 1, tf.compat.v1.shape(ifr_emb)[3]]) # [B, F, R, H]
		paddings = tf.compat.v1.zeros_like(ifr_emb) # [B, F, R, H]
		ifr_emb = tf.compat.v1.where(key_masks, ifr_emb, paddings)  # [B, F, R, H]
		
		#--------------social influence-------------------

		uid_emb_exp1 = tf.compat.v1.tile(uid_emb, [1, tf.compat.v1.shape(fuid_emb)[1]+1])
		uid_emb_exp1 = tf.compat.v1.reshape(uid_emb_exp1, [-1, tf.compat.v1.shape(fuid_emb)[1]+1, hidden_units_u]) # [B, F, H]
		iid_emb_exp1 = tf.compat.v1.tile(iid_emb, [1, tf.compat.v1.shape(fiid_emb)[1]+1])
		iid_emb_exp1 = tf.compat.v1.reshape(iid_emb_exp1, [-1, tf.compat.v1.shape(fiid_emb)[1]+1, hidden_units_i]) # [B, F, H]
		uid_emb_ = tf.compat.v1.expand_dims(uid_emb, axis = 1)
		iid_emb_ = tf.compat.v1.expand_dims(iid_emb, axis = 1)

		# GAT1: graph convolution on user's embedding for user static preference
		uid_in = tf.compat.v1.layers.dense(uid_emb_exp1, hidden_units_u, use_bias = False, name = 'trans_uid')
		fuid_in = tf.compat.v1.layers.dense(tf.compat.v1.concat([uid_emb_, fuid_emb], axis = 1), hidden_units_u, use_bias = False, reuse=True, name = 'trans_uid')
		din_gat_uid = tf.compat.v1.concat([uid_in, fuid_in], axis = -1)
		d1_gat_uid = tf.compat.v1.layers.dense(din_gat_uid, 1, activation=tf.compat.v1.nn.leaky_relu, name='gat_uid')
		d1_gat_uid = tf.compat.v1.nn.dropout(d1_gat_uid, keep_prob=self.keep_prob)
		d1_gat_uid = tf.compat.v1.reshape(d1_gat_uid, [-1, tf.compat.v1.shape(ufr_emb)[1]+1, 1]) # [B, F, 1]
		weights_uid = tf.compat.v1.nn.softmax(d1_gat_uid, axis=1)  # [B, F, 1]
		weights_uid = tf.compat.v1.tile(weights_uid, [1, 1, hidden_units_u]) # [B, F, H]
		uid_gat = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(weights_uid, fuid_in), axis = 1)
		uid_gat = tf.compat.v1.reshape(uid_gat, [-1, hidden_units_u])
		
		# GAT2: graph convolution on item's embedding for item static attribute
		iid_in = tf.compat.v1.layers.dense(iid_emb_exp1, hidden_units_i, use_bias = False, name = 'trans_iid')
		fiid_in = tf.compat.v1.layers.dense(tf.compat.v1.concat([iid_emb_, fiid_emb], axis = 1), hidden_units_i, use_bias = False, reuse=True, name = 'trans_iid')
		din_gat_iid = tf.compat.v1.concat([iid_in, fiid_in], axis = -1)
		d1_gat_iid = tf.compat.v1.layers.dense(din_gat_iid, 1, activation=tf.compat.v1.nn.leaky_relu, name='gat_iid')
		d1_gat_iid = tf.compat.v1.nn.dropout(d1_gat_iid, keep_prob=self.keep_prob)
		d1_gat_iid = tf.compat.v1.reshape(d1_gat_iid, [-1, tf.compat.v1.shape(ifr_emb)[1]+1, 1]) # [B, F, 1]
		weights_iid = tf.compat.v1.nn.softmax(d1_gat_iid, axis=1)  # [B, F, 1]
		weights_iid = tf.compat.v1.tile(weights_iid, [1, 1, hidden_units_i]) # [B, F, H]
		iid_gat = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(weights_iid, fiid_in), axis = 1)
		iid_gat = tf.compat.v1.reshape(iid_gat, [-1, hidden_units_i])


		uid_emb_exp2 = tf.compat.v1.tile(uid_emb, [1, tf.compat.v1.shape(ir_emb)[1]])
		uid_emb_exp2 = tf.compat.v1.reshape(uid_emb_exp2, [-1, tf.compat.v1.shape(ir_emb)[1], hidden_units_u]) # [B, R, H]
		iid_emb_exp2 = tf.compat.v1.tile(iid_emb, [1, tf.compat.v1.shape(ur_emb)[1]])
		iid_emb_exp2 = tf.compat.v1.reshape(iid_emb_exp2, [-1, tf.compat.v1.shape(ur_emb)[1], hidden_units_i]) # [B, R, H]
		ur_emb_ = tf.compat.v1.expand_dims(ur_emb, axis = 1) # [B, 1, R, H]
		ir_emb_ = tf.compat.v1.expand_dims(ir_emb, axis = 1) # [B, 1, R, H]
		uid_emb_exp3 = tf.compat.v1.expand_dims(uid_emb, axis = 1)
		uid_emb_exp3 = tf.compat.v1.expand_dims(uid_emb_exp3, axis = 2) # [B, 1, 1, H]
		uid_emb_exp3 = tf.compat.v1.tile(uid_emb_exp3, [1, tf.compat.v1.shape(ifr_emb)[1], tf.compat.v1.shape(ifr_emb)[2], 1]) # [B, F, R, H]
		iid_emb_exp3 = tf.compat.v1.expand_dims(iid_emb, axis = 1)
		iid_emb_exp3 = tf.compat.v1.expand_dims(iid_emb_exp3, axis = 2) # [B, 1, 1, H]
		iid_emb_exp3 = tf.compat.v1.tile(iid_emb_exp3, [1, tf.compat.v1.shape(ufr_emb)[1], tf.compat.v1.shape(ufr_emb)[2], 1]) # [B, F, R, H]

		# GAT3: graph convolution on user's clicked items for user dynamic preference
		uint_in = tf.compat.v1.multiply(ur_emb, iid_emb_exp2) # [B, R, H]
		uint_in = tf.compat.v1.reduce_max(uint_in, axis = 1) # [B, H]
		uint_in = tf.compat.v1.layers.dense(uint_in, hidden_units_i, use_bias = False, name = 'trans_uint') # [B, H]
		uint_in_ = tf.compat.v1.expand_dims(uint_in, axis = 1) # [B, 1, H]
		uint_in = tf.compat.v1.tile(uint_in, [1, tf.compat.v1.shape(ufr_emb)[1]+1])
		uint_in = tf.compat.v1.reshape(uint_in, [-1, tf.compat.v1.shape(ufr_emb)[1]+1, hidden_units_i]) # [B, F, H]
		fint_in = tf.compat.v1.multiply(ufr_emb, iid_emb_exp3) # [B, F, R, H]
		fint_in = tf.compat.v1.reduce_max(fint_in, axis = 2) # [B, F, H]
		fint_in = tf.compat.v1.layers.dense(fint_in, hidden_units_i, use_bias = False, reuse = True, name = 'trans_uint')
		fint_in = tf.compat.v1.concat([uint_in_, fint_in], axis = 1) # [B, F, H]
		din_gat_uint = tf.compat.v1.concat([uint_in, fint_in], axis = -1)
		d1_gat_uint = tf.compat.v1.layers.dense(din_gat_uint, 1, activation=tf.compat.v1.nn.leaky_relu, name='gat_uint')
		d1_gat_uint = tf.compat.v1.nn.dropout(d1_gat_uint, keep_prob=self.keep_prob)
		d1_gat_uint = tf.compat.v1.reshape(d1_gat_uint, [-1, tf.compat.v1.shape(ufr_emb)[1]+1, 1]) # [B, F, 1]
		weights_uint = tf.compat.v1.nn.softmax(d1_gat_uint, axis=1)  # [B, F, 1]
		weights_uint = tf.compat.v1.tile(weights_uint, [1, 1, hidden_units_i]) # [B, F, H]
		uint_gat = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(weights_uint, fint_in), axis = 1)
		uint_gat = tf.compat.v1.reshape(uint_gat, [-1, hidden_units_i])

		# GAT4: graph convolution on item's clicking users for item dynamic attribute
		iinf_in = tf.compat.v1.multiply(ir_emb, uid_emb_exp2) # [B, R, H]
		iinf_in = tf.compat.v1.reduce_max(iinf_in, axis = 1) # [B, H]
		iinf_in = tf.compat.v1.layers.dense(iinf_in, hidden_units_u, use_bias = False, name = 'trans_iinf') # [B, H]
		iinf_in_ = tf.compat.v1.expand_dims(iinf_in, axis = 1) # [B, 1, H]
		iinf_in = tf.compat.v1.tile(iinf_in, [1, tf.compat.v1.shape(ifr_emb)[1]+1])
		iinf_in = tf.compat.v1.reshape(iinf_in, [-1, tf.compat.v1.shape(ifr_emb)[1]+1, hidden_units_u]) # [B, F, H]
		finf_in = tf.compat.v1.multiply(ifr_emb, uid_emb_exp3) # [B, F, R, H]
		finf_in = tf.compat.v1.reduce_max(finf_in, axis = 2) # [B, F, H]
		finf_in = tf.compat.v1.layers.dense(finf_in, hidden_units_u, use_bias = False, reuse = True, name = 'trans_iinf')
		finf_in = tf.compat.v1.concat([iinf_in_, finf_in], axis = 1) # [B, F, H]
		din_gat_iinf = tf.compat.v1.concat([iinf_in, finf_in], axis = -1)
		d1_gat_iinf = tf.compat.v1.layers.dense(din_gat_iinf, 1, activation=tf.compat.v1.nn.leaky_relu, name='gat_iinf')
		d1_gat_iinf = tf.compat.v1.nn.dropout(d1_gat_iinf, keep_prob=self.keep_prob)
		d1_gat_iinf = tf.compat.v1.reshape(d1_gat_iinf, [-1, tf.compat.v1.shape(ifr_emb)[1]+1, 1]) # [B, F, 1]
		weights_iinf = tf.compat.v1.nn.softmax(d1_gat_iinf, axis=1)  # [B, F, 1]
		weights_iinf = tf.compat.v1.tile(weights_iinf, [1, 1, hidden_units_u]) # [B, F, H]
		iinf_gat = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(weights_iinf, finf_in), axis = 1)
		iinf_gat = tf.compat.v1.reshape(iinf_gat, [-1, hidden_units_u])

		#--------------DNN-based pairwise neural interaction layer---------------
		
		din_ui = tf.compat.v1.multiply(uid_gat, iid_gat)
		if self.training is True: 
			din_ui = tf.compat.v1.layers.batch_normalization(inputs=din_ui, name='norm_ui_b1', training = True)
		else:
			din_ui = tf.compat.v1.layers.batch_normalization(inputs=din_ui, name='norm_ui_b1', training = False)
		d1_ui = tf.compat.v1.layers.dense(din_ui, 16, activation=tf.compat.v1.nn.tanh, use_bias = True, name='norm_ui_1')
		d2_ui = tf.compat.v1.nn.dropout(d1_ui, keep_prob=self.keep_prob)
		d2_ui = tf.compat.v1.layers.dense(d2_ui, 8, activation=tf.compat.v1.nn.tanh, use_bias = True, name='norm_ui_2')
		d3_ui = tf.compat.v1.nn.dropout(d2_ui, keep_prob=self.keep_prob)
		d3_ui = tf.compat.v1.layers.dense(d3_ui, 4, activation=tf.compat.v1.nn.tanh, use_bias = True, name='norm_ui_3')
		d4_ui = tf.compat.v1.layers.dense(d3_ui, 1, activation=None, use_bias = True, name='norm_merge', reuse=tf.compat.v1.AUTO_REUSE)
		d4_ui = tf.compat.v1.reshape(d4_ui, [-1, 1])
		d3_ui_ = tf.compat.v1.reshape(d3_ui, [-1, tf.compat.v1.shape(d3_ui)[-1], 1])

		din_uf = tf.compat.v1.multiply(uid_gat, iinf_gat)
		if self.training is True:
			din_uf = tf.compat.v1.layers.batch_normalization(inputs=din_uf, name='norm_uf_b1', training = True)
		else:
			din_uf = tf.compat.v1.layers.batch_normalization(inputs=din_uf, name='norm_uf_b1', training = False)
		d1_uf = tf.compat.v1.layers.dense(din_uf, 16, activation=tf.compat.v1.nn.tanh, use_bias = True, name='norm_uf_1')
		d2_uf = tf.compat.v1.nn.dropout(d1_uf, keep_prob=self.keep_prob)
		d2_uf = tf.compat.v1.layers.dense(d2_uf, 8, activation=tf.compat.v1.nn.tanh, use_bias = True, name='norm_uf_2')
		d3_uf = tf.compat.v1.nn.dropout(d2_uf, keep_prob=self.keep_prob)
		d3_uf = tf.compat.v1.layers.dense(d3_uf, 4, activation=tf.compat.v1.nn.tanh, use_bias = True, name='norm_uf_3')
		d4_uf = tf.compat.v1.layers.dense(d3_uf, 1, activation=None, use_bias = True, name='norm_merge', reuse=tf.compat.v1.AUTO_REUSE)
		d4_uf = tf.compat.v1.reshape(d4_uf, [-1, 1])
		d3_uf_ = tf.compat.v1.reshape(d3_uf, [-1, tf.compat.v1.shape(d3_uf)[-1], 1])

		din_fi = tf.compat.v1.multiply(uint_gat, iid_gat)
		if self.training is True:
			din_fi = tf.compat.v1.layers.batch_normalization(inputs=din_fi, name='norm_fi_b1', training = True)
		else:
			din_fi = tf.compat.v1.layers.batch_normalization(inputs=din_fi, name='norm_fi_b1', training = False)
		d1_fi = tf.compat.v1.layers.dense(din_fi, 16, activation=tf.compat.v1.nn.tanh, use_bias = True, name='norm_fi_1')
		d2_fi = tf.compat.v1.nn.dropout(d1_fi, keep_prob=self.keep_prob)
		d2_fi = tf.compat.v1.layers.dense(d2_fi, 8, activation=tf.compat.v1.nn.tanh, use_bias = True, name='norm_fi_2')
		d3_fi = tf.compat.v1.nn.dropout(d2_fi, keep_prob=self.keep_prob)
		d3_fi = tf.compat.v1.layers.dense(d3_fi, 4, activation=tf.compat.v1.nn.tanh, use_bias = True, name='norm_fi_3')
		d4_fi = tf.compat.v1.layers.dense(d3_fi, 1, activation=None, use_bias = True, name='norm_merge', reuse=tf.compat.v1.AUTO_REUSE)
		d4_fi = tf.compat.v1.reshape(d4_fi, [-1, 1])
		d3_fi_ = tf.compat.v1.reshape(d3_fi, [-1, tf.compat.v1.shape(d3_fi)[-1], 1])

		din_ff = tf.compat.v1.multiply(uint_gat, iinf_gat)
		if self.training is True:
			din_ff = tf.compat.v1.layers.batch_normalization(inputs=din_ff, name='norm_ff_b1', training = True)
		else:
			din_ff = tf.compat.v1.layers.batch_normalization(inputs=din_ff, name='norm_ff_b1', training = False)
		d1_ff = tf.compat.v1.layers.dense(din_ff, 16, activation=tf.compat.v1.nn.tanh, use_bias = True, name='norm_ff_1')
		d2_ff = tf.compat.v1.nn.dropout(d1_ff, keep_prob=self.keep_prob)
		d2_ff = tf.compat.v1.layers.dense(d2_ff, 8, activation=tf.compat.v1.nn.tanh, use_bias = True, name='norm_ff_2')
		d3_ff = tf.compat.v1.nn.dropout(d2_ff, keep_prob=self.keep_prob)
		d3_ff = tf.compat.v1.layers.dense(d3_ff, 4, activation=tf.compat.v1.nn.tanh, use_bias = True, name='norm_ff_3')
		d4_ff = tf.compat.v1.layers.dense(d3_ff, 1, activation=None, use_bias = True, name='norm_merge', reuse=tf.compat.v1.AUTO_REUSE)
		d4_ff = tf.compat.v1.reshape(d4_ff, [-1, 1])
		d3_ff_ = tf.compat.v1.reshape(d3_ff, [-1, tf.compat.v1.shape(d3_ff)[-1], 1])

		d3 = tf.compat.v1.concat([d3_ui_, d3_uf_, d3_fi_, d3_ff_], axis = 2)

		#--------------policy-based fusion layer---------------
		def policy(uid_emb, iid_emb, l_name = 'policy_1'):
			din_policy = tf.compat.v1.concat([uid_emb, iid_emb, tf.compat.v1.multiply(uid_emb, iid_emb)], axis = -1)
			policy = tf.compat.v1.layers.dense(din_policy, 4, activation=None, name=l_name)
			policy = tf.compat.v1.nn.softmax(policy)
			return policy

		policy1 = policy(uid_emb, iid_emb, 'policy_1')
		policy2 = policy(uid_emb, iid_emb, 'policy_2')
		policy3 = policy(uid_emb, iid_emb, 'policy_3')
		policy4 = policy(uid_emb, iid_emb, 'policy_4')
		policy = (policy1 + policy2 + policy3 + policy4) / 4
		policy_exp = tf.compat.v1.tile(policy, [1, tf.compat.v1.shape(d3_ui)[-1]])
		policy_exp = tf.compat.v1.reshape(policy_exp, [-1, tf.compat.v1.shape(d3_ui)[-1], 4])
		if self.training == True:
			dist = tf.compat.v1.distributions.Multinomial(total_count = 1., probs = policy)
			t = dist.sample(1)
			t = tf.compat.v1.reshape(t, [-1, 4]) #[B, 4]
			t_exp = tf.compat.v1.tile(t, [1, tf.compat.v1.shape(d3_ui)[-1]])
			t_exp = tf.compat.v1.reshape(t_exp, [-1, tf.compat.v1.shape(d3_ui)[-1], 4])
			dmerge = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(t_exp, d3), axis = 2)
		else:
			dmerge = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(policy_exp, d3), axis = 2)
		dmerge = tf.compat.v1.reshape(dmerge, [-1, 4])
		dmerge = tf.compat.v1.layers.dense(dmerge, 1, activation=None, use_bias = True, name='norm_merge', reuse=tf.compat.v1.AUTO_REUSE)
		dmerge = tf.compat.v1.reshape(dmerge, [-1])

		#--------------output layer---------------
		self.logits = i_b + dmerge
		self.score = self.logits
		i_b_exp = tf.compat.v1.reshape(i_b, [-1, 1])
		logits_policy = tf.compat.v1.concat([i_b_exp + d4_ui, i_b_exp + d4_uf, i_b_exp + d4_fi, i_b_exp + d4_ff], axis = -1)
		score_policy = logits_policy

		# loss function
		loss_emb_reg = tf.compat.v1.reduce_sum(tf.compat.v1.abs(i_b)) + tf.compat.v1.reduce_sum(tf.compat.v1.abs(iid_emb)) + tf.compat.v1.reduce_sum(tf.compat.v1.abs(uid_emb)) + tf.compat.v1.reduce_sum(tf.compat.v1.abs(fuid_emb))
		self.loss = tf.compat.v1.reduce_mean(tf.compat.v1.square(self.score-self.label)) + self.lambda1*loss_emb_reg

		# loss for each policy net
		labels_exp = tf.compat.v1.reshape(self.label, [-1, 1])
		self.loss_p1 = tf.compat.v1.reduce_mean(tf.compat.v1.reduce_sum(tf.compat.v1.multiply(-tf.compat.v1.log(policy1), -tf.compat.v1.square(score_policy-labels_exp)), axis = -1))
		self.loss_p2 = tf.compat.v1.reduce_mean(tf.compat.v1.reduce_sum(tf.compat.v1.multiply(-tf.compat.v1.log(policy2), -tf.compat.v1.square(score_policy-labels_exp)), axis = -1))
		self.loss_p3 = tf.compat.v1.reduce_mean(tf.compat.v1.reduce_sum(tf.compat.v1.multiply(-tf.compat.v1.log(policy3), -tf.compat.v1.square(score_policy-labels_exp)), axis = -1))
		self.loss_p4 = tf.compat.v1.reduce_mean(tf.compat.v1.reduce_sum(tf.compat.v1.multiply(-tf.compat.v1.log(policy4), -tf.compat.v1.square(score_policy-labels_exp)), axis = -1))

		# Step variable
		self.global_step = tf.compat.v1.Variable(0, trainable=False, name='global_step')
		self.global_epoch_step = \
		tf.compat.v1.Variable(0, trainable=False, name='global_epoch_step')
		self.global_epoch_step_op = \
		tf.compat.v1.assign(self.global_epoch_step, self.global_epoch_step+1)

		# optimization for loss function
		self.opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
		trainable_params = tf.compat.v1.trainable_variables(scope = 'norm')
		gradients = tf.compat.v1.gradients(self.loss, trainable_params)
		clip_gradients, _ = tf.compat.v1.clip_by_global_norm(gradients, 5*self.learning_rate)
		self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)
		
		# optimization for each policy net
		trainable_params1 = tf.compat.v1.trainable_variables(scope = 'policy_1')
		gradients1 = tf.compat.v1.gradients(self.loss_p1, trainable_params1)
		clip_gradients1, _ = tf.compat.v1.clip_by_global_norm(gradients1, 5*self.learning_rate)
		self.train_op1 = self.opt.apply_gradients(zip(clip_gradients1, trainable_params1))

		trainable_params2 = tf.compat.v1.trainable_variables(scope = 'policy_2')
		gradients2 = tf.compat.v1.gradients(self.loss_p2, trainable_params2)
		clip_gradients2, _ = tf.compat.v1.clip_by_global_norm(gradients2, 5*self.learning_rate)
		self.train_op2 = self.opt.apply_gradients(zip(clip_gradients2, trainable_params2))

		trainable_params3 = tf.compat.v1.trainable_variables(scope = 'policy_3')
		gradients3 = tf.compat.v1.gradients(self.loss_p3, trainable_params3)
		clip_gradients3, _ = tf.compat.v1.clip_by_global_norm(gradients3, 5*self.learning_rate)
		self.train_op3 = self.opt.apply_gradients(zip(clip_gradients3, trainable_params3))

		trainable_params4 = tf.compat.v1.trainable_variables(scope = 'policy_4')
		gradients4 = tf.compat.v1.gradients(self.loss_p4, trainable_params4)
		clip_gradients4, _ = tf.compat.v1.clip_by_global_norm(gradients4, 5*self.learning_rate)
		self.train_op4 = self.opt.apply_gradients(zip(clip_gradients4, trainable_params4))
		
		#--------------end model---------------

	def train(self, sess, datainput, u_readinput, u_friendinput, uf_readinput, u_read_l, u_friend_l, uf_read_l, i_readinput, i_friendinput, if_readinput, i_linkinput, i_read_l, i_friend_l, if_read_l, lr, keep_prob, lambda1, lambda2):
		loss, _ = sess.run([self.loss, self.train_op], feed_dict={
		self.item: datainput[0], self.user: datainput[1], self.label: datainput[2], \
		self.u_read: u_readinput, self.u_friend: u_friendinput, self.uf_read: uf_readinput, self.u_read_l: u_read_l, self.u_friend_l: u_friend_l, self.uf_read_l: uf_read_l,
		self.i_read: i_readinput, self.i_friend: i_friendinput, self.if_read: if_readinput, self.i_link: i_linkinput, self.i_read_l: i_read_l, self.i_friend_l: i_friend_l, self.if_read_l: if_read_l,
		self.training: 1, self.learning_rate: lr, self.keep_prob: keep_prob, self.lambda1: lambda1, self.lambda2: lambda2,
		})
		return loss
	
	def policy_update(self, sess, datainput, u_readinput, u_friendinput, uf_readinput, u_read_l, u_friend_l, uf_read_l, i_readinput, i_friendinput, if_readinput, i_linkinput, i_read_l, i_friend_l, if_read_l, lr, keep_prob, lambda1, lambda2):
		_ = sess.run([self.train_op1], feed_dict={
		self.item: datainput[0], self.user: datainput[1], self.label: datainput[2], \
		self.u_read: u_readinput, self.u_friend: u_friendinput, self.uf_read: uf_readinput, self.u_read_l: u_read_l, self.u_friend_l: u_friend_l, self.uf_read_l: uf_read_l,
		self.i_read: i_readinput, self.i_friend: i_friendinput, self.if_read: if_readinput, self.i_link: i_linkinput, self.i_read_l: i_read_l, self.i_friend_l: i_friend_l, self.if_read_l: if_read_l,
		self.training: 1, self.learning_rate: lr, self.keep_prob: keep_prob, self.lambda1: lambda1, self.lambda2: lambda2,
		})
		_ = sess.run([self.train_op2], feed_dict={
		self.item: datainput[0], self.user: datainput[1], self.label: datainput[2], \
		self.u_read: u_readinput, self.u_friend: u_friendinput, self.uf_read: uf_readinput, self.u_read_l: u_read_l, self.u_friend_l: u_friend_l, self.uf_read_l: uf_read_l,
		self.i_read: i_readinput, self.i_friend: i_friendinput, self.if_read: if_readinput, self.i_link: i_linkinput, self.i_read_l: i_read_l, self.i_friend_l: i_friend_l, self.if_read_l: if_read_l,
		self.training: 1, self.learning_rate: lr, self.keep_prob: keep_prob, self.lambda1: lambda1, self.lambda2: lambda2,
		})
		_ = sess.run([self.train_op3], feed_dict={
		self.item: datainput[0], self.user: datainput[1], self.label: datainput[2], \
		self.u_read: u_readinput, self.u_friend: u_friendinput, self.uf_read: uf_readinput, self.u_read_l: u_read_l, self.u_friend_l: u_friend_l, self.uf_read_l: uf_read_l,
		self.i_read: i_readinput, self.i_friend: i_friendinput, self.if_read: if_readinput, self.i_link: i_linkinput, self.i_read_l: i_read_l, self.i_friend_l: i_friend_l, self.if_read_l: if_read_l,
		self.training: 1, self.learning_rate: lr, self.keep_prob: keep_prob, self.lambda1: lambda1, self.lambda2: lambda2,
		})
		_ = sess.run([self.train_op4], feed_dict={
		self.item: datainput[0], self.user: datainput[1], self.label: datainput[2], \
		self.u_read: u_readinput, self.u_friend: u_friendinput, self.uf_read: uf_readinput, self.u_read_l: u_read_l, self.u_friend_l: u_friend_l, self.uf_read_l: uf_read_l,
		self.i_read: i_readinput, self.i_friend: i_friendinput, self.if_read: if_readinput, self.i_link: i_linkinput, self.i_read_l: i_read_l, self.i_friend_l: i_friend_l, self.if_read_l: if_read_l,
		self.training: 1, self.learning_rate: lr, self.keep_prob: keep_prob, self.lambda1: lambda1, self.lambda2: lambda2,
		})

	def eval(self, sess, datainput, u_readinput, u_friendinput, uf_readinput, u_read_l, u_friend_l, uf_read_l, i_readinput, i_friendinput, if_readinput, i_linkinput, i_read_l, i_friend_l, if_read_l, lambda1, lambda2):
		score, loss = sess.run([self.score, self.loss], feed_dict={
		self.item: datainput[0], self.user: datainput[1], self.label: datainput[2], \
		self.u_read: u_readinput, self.u_friend: u_friendinput, self.uf_read: uf_readinput, self.u_read_l: u_read_l, self.u_friend_l: u_friend_l, self.uf_read_l: uf_read_l,
		self.i_read: i_readinput, self.i_friend: i_friendinput, self.if_read: if_readinput, self.i_link: i_linkinput, self.i_read_l: i_read_l, self.i_friend_l: i_friend_l, self.if_read_l: if_read_l,
		self.training: 0, self.keep_prob: 1, self.lambda1: lambda1, self.lambda2: lambda2,
		})
		return score, loss

	def save(self, sess, path):
		saver = tf.compat.v1.train.Saver()
		saver.save(sess, save_path=path)

	def restore(self, sess, path):
		saver = tf.compat.v1.train.Saver()
		saver.restore(sess, save_path=path)
