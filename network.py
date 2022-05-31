import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from einops.layers.tensorflow import Rearrange

tf.autograph.experimental.do_not_convert
def get_ssl_network(signal_length, segment_size, signal_channels,  code_size=64, l2_rate=1e-4):

  encoder = SensorTransformer(
    signal_length=signal_length,
    segment_size=segment_size,
    channels=signal_channels,
    num_layers=4,
    d_model=64,
    num_heads=4,
    mlp_dim=64,
    dropout=0.1)

  inputs = tf.keras.layers.Input(
    (signal_length, signal_channels))
  x = encoder(inputs)
  x  = tf.keras.layers.GlobalMaxPooling1D()(x)
  x = tf.keras.layers.Dense(code_size, activation="linear", 
    kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(x)
  x = tf.keras.layers.LayerNormalization()(x)
  outputs = tf.keras.layers.Activation("tanh")(x)
  embedding_model = tf.keras.Model(inputs, outputs, 
    name = "embedding_model")
  similarity_layer = BilinearProduct(code_size)
  return ContrastiveModel(embedding_model, similarity_layer)


class TransformerBlock(tf.keras.layers.Layer):
  def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
    super(TransformerBlock, self).__init__()
    self.att = tf.keras.layers.MultiHeadAttention(
      num_heads=num_heads, 
      key_dim=embed_dim) 
    self.ffn = tf.keras.Sequential(
        [tf.keras.layers.Dense(ff_dim, activation=tf.keras.activations.gelu), 
        tf.keras.layers.Dense(embed_dim)]
    )
    self.layernorm_a = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm_b = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout_a = tf.keras.layers.Dropout(dropout)
    self.dropout_b = tf.keras.layers.Dropout(dropout)

  def call(self, inputs, training):
    attn_output = self.att(inputs, inputs)
    attn_output = self.dropout_a(attn_output, 
      training=training)
    out_a = self.layernorm_a(inputs + attn_output)
    ffn_output = self.ffn(out_a)
    ffn_output = self.dropout_b(ffn_output, 
      training=training)
    return self.layernorm_b(out_a + ffn_output)

class SensorTransformer(tf.keras.Model):
    def __init__(
      self,
      signal_length,
      segment_size,
      channels,
      num_layers,
      d_model,
      num_heads,
      mlp_dim,
      dropout):
      super(SensorTransformer, self).__init__()
      num_patches = (signal_length // segment_size)

      self.segment_size = segment_size
      self.d_model = d_model
      self.num_layers = num_layers

      self.pos_emb = self.add_weight("pos_emb", 
        shape=(1, num_patches + 1, d_model))
      self.class_emb = self.add_weight("class_emb", 
        shape=(1, 1, d_model))
      self.patch_proj = tf.keras.layers.Dense(d_model)
      self.enc_layers = [
        TransformerBlock(d_model, num_heads, mlp_dim, dropout)
        for _ in range(num_layers)
      ]

      self.inst_norm = tfa.layers.InstanceNormalization(axis=2, 
            epsilon=1e-6,
            center=False, 
            scale=False, 
            beta_initializer="glorot_uniform",
            gamma_initializer="glorot_uniform")

    def call(self, input, training):        
      batch_size = tf.shape(input)[0]
      #x = self.inst_norm(input)
      patches = Rearrange("b (w p1) c-> b w (p1 c)", 
        p1=self.segment_size)(input) #(x) #(x)
      x = self.patch_proj(patches)

      class_emb = tf.broadcast_to(self.class_emb, 
        [batch_size, 1, self.d_model])
      
      x = tf.concat([class_emb, x], axis=1)
      x = x + self.pos_emb
      
      for layer in self.enc_layers:
        x = layer(x, training)

      return x

class BilinearProduct(tf.keras.layers.Layer):
  """Bilinear product."""
  def __init__(self, dim):
    super(BilinearProduct, self).__init__()
    self.dim = dim

  def build(self, _):
    self._w = self.add_weight(shape=(self.dim, self.dim),
      initializer="glorot_uniform",
      trainable=True,
      name="bilinear_product_weight")

  def call(self, anchor, positive):
    projection_positive = tf.linalg.matmul(self._w, positive, transpose_b=True)
    return tf.linalg.matmul(anchor, projection_positive)

class ContrastiveModel(tf.keras.Model):
  """Wrapper class for custom contrastive model."""

  def __init__(self, embedding_model, similarity_layer):
    super().__init__()
    self.embedding_model = embedding_model
    self._similarity_layer = similarity_layer

  def train_step(self, data):
    anchors, positives = data

    with tf.GradientTape() as tape:
      inputs = tf.concat([anchors, positives], axis=0)
      embeddings = self.embedding_model(inputs, training=True)
      anchor_embeddings, positive_embeddings = tf.split(embeddings, 2, axis=0)
      similarities = self._similarity_layer(anchor_embeddings, positive_embeddings)
      sparse_labels = tf.range(tf.shape(anchors)[0])
      loss = self.compiled_loss(sparse_labels, similarities)
      loss += sum(self.losses)

    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    self.compiled_metrics.update_state(sparse_labels, similarities)
    return {m.name: m.result() for m in self.metrics}

  def call(self, input, training=False):
    return self.embedding_model(input, training)