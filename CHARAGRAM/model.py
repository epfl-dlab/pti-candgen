import tensorflow as tf
import numpy as np

class Speller(tf.keras.Model):
    def __init__(self, mnt_size, ent_size, embed_size):
        super(Speller, self).__init__()
        self.mnt_size = mnt_size
        self.ent_size = ent_size
        self.embed_size = embed_size
        initializer = tf.initializers.GlorotUniform()
        
        self.mnt_matrix = tf.Variable(initializer((self.mnt_size, self.embed_size)).numpy(), dtype=np.float32, name="mnt_matrix")
        self.ent_matrix = tf.Variable(initializer((self.ent_size, self.embed_size)).numpy(), dtype=np.float32, name="ent_matrix")
        
        self.mnt_bias = tf.Variable(np.zeros([self.embed_size]), dtype=np.float32, name="mnt_bias")
        self.ent_bias = tf.Variable(np.zeros([self.embed_size]), dtype=np.float32, name="ent_bias")

    def call(self, mention_sp, ent_pos_sp, ent_neg_sp):
        mnt_embed = tf.nn.embedding_lookup_sparse(self.mnt_matrix, mention_sp, None, combiner="sum")
        mnt_embed = tf.math.tanh(tf.math.add(mnt_embed, self.mnt_bias))

        pos_ent_embed = tf.nn.embedding_lookup_sparse(self.ent_matrix, ent_pos_sp, None, combiner="sum")
        pos_ent_embed = tf.math.tanh(tf.math.add(pos_ent_embed, self.ent_bias))

        neg_ent_embed = tf.nn.embedding_lookup_sparse(self.ent_matrix, ent_neg_sp, None, combiner="sum")
        neg_ent_embed = tf.math.tanh(tf.math.add(neg_ent_embed, self.ent_bias))
        
        sim_mnt_pos = tf.keras.losses.cosine_similarity(mnt_embed, pos_ent_embed, axis=-1)
        sim_mnt_neg = tf.keras.losses.cosine_similarity(mnt_embed, neg_ent_embed, axis=-1)
        return sim_mnt_pos, sim_mnt_neg

    def mnt_rpr(self, mention_sp):
        mnt_embed = tf.nn.embedding_lookup_sparse(self.mnt_matrix, mention_sp, None, combiner="sum")
        mnt_embed = tf.math.tanh(tf.math.add(mnt_embed, self.mnt_bias))
        return mnt_embed

    def ent_rpr(self, ent_sp):
        ent_embed = tf.nn.embedding_lookup_sparse(self.ent_matrix, ent_sp, None, combiner="sum")
        ent_embed = tf.math.tanh(tf.math.add(ent_embed, self.ent_bias))
        return ent_embed
