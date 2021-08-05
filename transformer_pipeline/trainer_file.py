import tensorflow as tf
from tensorflow import keras

import numpy as np

import time

from typing import List, Text
import absl
import tensorflow_transform as tft

import tfx_bsl
from tfx_bsl.tfxio import dataset_options

import tfx
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.trainer.fn_args_utils import FnArgs

_lang1 = 'lang1'
_lang2 = 'lang2'
_batch_size = 128

def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example."""
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_lang2)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    transformed_features = model.tft_layer(parsed_features)
    return model(transformed_features)

  return serve_tf_examples_fn

#Defining the Pointwise Feed Forward Network for usage in both decoder and encoder layers
def pointwiseFFN(dimModel = 512, depthFeedForward = 2048):
  return tf.keras.Sequential(
      [
       tf.keras.layers.Dense(depthFeedForward, activation = 'relu'),
       tf.keras.layers.Dense(dimModel)
      ]
  )

#SECTION: POSITIONAL ENCODING
#Here, we make use of the formula as explained in the paper
def position_value(pos, d_model):
  angle_rates = 1 / np.power(10000, (2 * (pos//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(d_model, max_sen_len):

  positions = tf.linspace(0, max_sen_len - 1, max_sen_len).numpy()
  angles = [[position_value(i, d_model)] for i in positions]
  angle_rads = tf.ones([max_sen_len, d_model]).numpy()
  angle_rads[0::2, :] = angle_rads[0::2, :] * np.sin(angles[0::2])
  angle_rads[1::2, :] = angle_rads[1::2, :] * np.cos(angles[1::2])

  return tf.cast(angle_rads, dtype=tf.float32)

#SECTION: MASK
#WE build masks needed for hiding padded tokens, or future tokens while decoding
def lookahead_mask(token_list):
  mask = tf.cast(tf.math.equal(token_list, tf.constant(0, dtype = tf.float32)), tf.float32)
  mask = [[mask_val] for mask_val in mask]
  return tf.cast(mask, tf.float32)

#SECTION: MULTIHEADED ATTENTION
#We first build a single head for attention
class SingleHeadAttention(tf.keras.layers.Layer):
  def __init__(self, dimModel = 512):

    super(SingleHeadAttention, self).__init__()

    self.dimModel = dimModel
    self.wq = tf.keras.layers.Dense(dimModel)
    self.wk = tf.keras.layers.Dense(dimModel)
    self.wv = tf.keras.layers.Dense(dimModel)

  def call(self, embeddings_q, embeddings_k, embeddings_v, mask):
    q = self.wq(embeddings_q)
    k = self.wk(embeddings_k)
    v = self.wv(embeddings_v)

    matMul = tf.matmul(q, tf.transpose(k, perm = [0,2,1]))
    scaledMatMul = matMul/tf.math.sqrt(tf.cast(self.dimModel, tf.float32))

    if mask is not None:
      scaledMatMul = scaledMatMul + (mask * -1e9)

    attentionWeights = tf.nn.softmax(scaledMatMul, axis = -1)

    attentionScores = tf.matmul(attentionWeights, tf.cast(v, tf.float32))

    resultDict = {
        "q": q,
        "k": k,
        "v": v,
        "attentionWeights": attentionWeights,
        "attentionScores": attentionScores
    }

    return resultDict

#And then combine them all into the MultiHeadAttention, to be used in the encoder as well as decoder layers
class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, numHeads = 8, dimModel = 512):

    super(MultiHeadAttention, self).__init__()

    self.dimModel = dimModel
    self.numHeads = numHeads
    self.attentionHeads = [SingleHeadAttention(self.dimModel) for _ in range(self.numHeads)]

  def call(self, embeddings_q, embeddings_k, embeddings_v, mask):
    outputs = [attention(embeddings_q, embeddings_k, embeddings_v, mask) for attention in self.attentionHeads]
    output_k = [output["k"] for output in outputs]
    output_v = [output["v"] for output in outputs]


    output_scores = [result["attentionScores"] for result in outputs]

    concat_layer = tf.keras.layers.Concatenate()

    concatenated = concat_layer(output_scores)

    linear_layer = tf.keras.layers.Dense(self.dimModel)

    resultOutput = linear_layer(concatenated)

    resultDict = {
        "output": resultOutput,
        "output_k": output_k,
        "output_v": output_v
    }

    return resultDict

#SECTION: ENCODER LAYER
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, numHeads = 8, dimModel = 512, depthFeedForward = 2048, rate = 0.05):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(numHeads = numHeads, dimModel = dimModel)

    self.ffn = pointwiseFFN(dimModel, depthFeedForward)

    self.normLayer1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
    self.normLayer2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, embeddings, mask = None):

    embeddings = tf.cast(embeddings, tf.float32)

    output = self.mha(embeddings, embeddings, embeddings, mask = None)

    attn_output = output["output"]
    attn_output = self.dropout1(attn_output)
    output1 = self.normLayer1(embeddings + attn_output)

    outputFFN = self.ffn(output1)
    ffn_output = self.dropout2(outputFFN)
    outputFinal = self.normLayer2(outputFFN + output1)

    resultDict = {
        "output": outputFinal,
        "output_k": output["output_k"],
        "output_v": output["output_v"]
    }

    return resultDict

#SECTION: DECODER LAYER
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, numHeads = 8, dimModel = 512, depthFeedForward = 2048, rate = 0.05):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(numHeads = numHeads, dimModel = dimModel)
    self.mha2 = MultiHeadAttention(numHeads = numHeads, dimModel = dimModel)

    self.ffn = pointwiseFFN(dimModel, depthFeedForward)

    self.normLayer1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
    self.normLayer2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
    self.normLayer3 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, embeddings, encoder_k, encoder_v, mask):

    embeddings = tf.cast(embeddings, tf.float32)

    mha_output1 = self.mha1(embeddings, embeddings, embeddings, mask)
    attn_output1 = mha_output1["output"]
    attn_output1 = self.dropout1(attn_output1)
    output1 = self.normLayer1(embeddings + attn_output1)

    mha_output2 = self.mha2(output1, encoder_k, encoder_v, mask)
    attn_output2 = mha_output2["output"]
    attn_output2 = self.dropout2(attn_output2)
    output2 = self.normLayer2(attn_output1 + attn_output2)

    outputFFN = self.ffn(output2)
    ffn_output = self.dropout3(outputFFN)
    outputFinal = self.normLayer3(outputFFN + output2)

    return outputFinal

#SECTION: ENCODER STACK
#We build out a stack of the encoders using the individual encoder layers defined above
#This includes the embedding layers as well
#The output is sent to the Decoder Stack to help it generate its key and value pairs for MultiHeadAttention
class EncoderStack(tf.keras.layers.Layer):
  def __init__(self, vocab_size = 10000, numEncoders = 6, dimModel = 512, numHeads = 8, depthFeedForward = 2048, max_sen_len = 512):
    super(EncoderStack, self).__init__()
    self.numEncoders = numEncoders
    self.EncoderLayers = [EncoderLayer(dimModel = dimModel, numHeads = numHeads, depthFeedForward = depthFeedForward) for i in range(numEncoders)]
    self.embedding = tf.keras.layers.Embedding(vocab_size, dimModel)

    self.positional_encoding = positional_encoding(dimModel, max_sen_len)

  def call(self, tokenised_input):

    embedded = self.embedding(tokenised_input)
    encoded = embedded + self.positional_encoding

    x = encoded

    for i in range(self.numEncoders):
      output = self.EncoderLayers[i](x, mask = None)
      x = output["output"]


    resultDict = {
        "output": x,
    }

    return resultDict

#SECTION: ENCODER STACK
#We build out a stack of the encoders using the individual encoder layers defined above
#This includes the embedding layers as well
class DecoderStack(tf.keras.layers.Layer):
  def __init__(self, vocab_size = 10000, numDecoders = 6, dimModel = 512, numHeads = 8, depthFeedForward = 2048, max_sen_len = 512):
    super(DecoderStack, self).__init__()
    self.numDecoders = numDecoders
    self.DecoderLayers = [DecoderLayer(dimModel = dimModel, numHeads = numHeads, depthFeedForward = depthFeedForward) for i in range(numDecoders)]
    self.embedding = tf.keras.layers.Embedding(vocab_size, dimModel)

    self.positional_encoding = positional_encoding(dimModel, max_sen_len)

  def call(self, tokenised_input, encoder_output, mask):

    embedded = self.embedding(tokenised_input)
    encoded = embedded + self.positional_encoding
    x = encoded

    for i in range(self.numDecoders):
      x = self.DecoderLayers[i](x, encoder_output, encoder_output, mask)

    return x

#SECTION: TRANSFORMER
class Transformer(tf.keras.Model):
  def __init__(self, tar_vocab_size = 10000, numEncoders = 6, numDecoders = 6, dimModel = 512, depthFeedForward = 2048, start_token = '<START>', end_token = '<END>', max_sen_len = 512):
    self.encoder = EncoderStack(vocab_size = vocab_size, numEncoders = numEncoders, dimModel = dimModel, depthFeedForward = depthFeedForward)
    self.decoder = DecoderStack(vocab_size = vocab_size, numDecoders = numDecoders, dimModel = dimModel, depthFeedForward = depthFeedForward)
    self.linearLayer = tf.keras.layers.Dense(tar_vocab_size)

  def call(self, inpTokens):

    encoded = self.EncoderStack(inpTokens)
    encoded_output = encoded['output']

    def pad_to_max_len(sentence, numDecoded, max_sen_len):
      cur_sentence = sentence[0:numDecoded]
      paddings = tf.constant([[0,0], [0, max_sen_len - numDecoded]])
      return tf.pad(tensor = cur_sentence, paddings = paddings, mode = "CONSTANT", constant_values = 0)

    start = findToken(start_token)
    end = findToken(end_token)

    predicted_sentence = [[start]]
    decoded_steps = 0

    outputs = tf.constant([])

    #This is the section where we predict outputs one step at a time
    #We start with the <START> token, and keep using it to predict the next token,
    #and don't stop until we hit <END> token
    #Here, we train it the traditional way, but the official TF docs also provide an example
    #for how to use teacher forcing in the training
    while ((pred != end) and (decoded_steps < max_sen_len - 1)):
      padded_sentence = pad_to_max_len(tf.reshape(tf.constant(predicted_sentence, tf.float32), [-1]), decoded_steps, self.max_sen_len)
      mask = lookahead_mask(predicted_sentence)
      output = self.DecoderStack(with_pad_token, mask)
      linearOutput = self.linearLayer(output)
      logits = tf.nn.softmax(linearOutput)

      decoded_steps = decoded_steps + 1

      if decoded_steps == max_sen_len - 1:
        pred = end

      predicted_sentence = predicted_sentence.append([pred])

    return outputs

#Needs a custom rate scheduler, have used paper definition
class learning_rate(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, dimModel, warmup_steps=4000):
    super(learning_rate, self).__init__()

    self.dimModel = dimModel
    self.dimModel = tf.cast(self.dimModel, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

#Define loss
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

#Since we cannot technically use all tokens in the sentence, which are max_sen_len
#but not every sentence will be of length max_sen_len, we need to mask out those
#particular tokens that aren't a part of the sentence and then find the Cross Entropy loss
def loss_func(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_val = loss(real, pred)
  loss_val = loss_val * mask
  return tf.reduce_sum(loss_val)/tf.reduce_sum(mask)

#Define Accuracy
def accuracy(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

#Building Transformer with all the default values
def build_model():
    return Transformer()

#Define one step of the training
def train_step(inp, tar):
  with tf.GradientTape as tape:
    predicted_logits = transformer(inp)
    loss = loss_func(predicted_logits, tar)
  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss(loss)
  train_accuracy(accuracy(tar_real, predictions))

#Define data access function
def _input_fn(
    file_pattern: List[Text],
    data_accessor: DataAccessor,
    tf_transform_output : tft.TFTransformOutput,
    batch_size: int = _batch_size):

    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(batch_size = _batch_size, label_key = 'hin_xf'),
        tf_transform_output.transformed_metadata.schema
    )

    return dataset

#Define how training will occur
def run_training(epochs = 10, train_dataset, transformer):
  for epoch in range epochs:

    train_loss.reset_states()
    train_accuracy.reset_states()

    start = time.time()

    for (batch, (inp, tar)) in enumerate(train_dataset):
      train_step(inp, tar, transformer)
    tf.print(f'Epoch {epoch+1}: Time Taken - {time.time() - start}, Accuracy - {train_accuracy.result()}, Loss - {train_loss.result()}')

def run_fn(fn_args: FnArgs):

    NUM_EPOCHS = 10
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    tft_layer = tf_transform_output.transform_features_layer()

    #TFT does not provide us a Tokenize layer.
    #Instead, it builds a vocabulary file (default in txt format) that can be found at the location vocab_hi_loc
    #If you do not only want to produce the logits, but also want to predict the tokens,
    #you can use the below lists
    vocab_hi_loc = fn_args.transform_graph_path + '/transform_fn/assets/vocab_hi'

    vocab_hi = []

    with open(vocab_hi_loc) as text:
        vocab_hi = text.readlines()

    #Performing str.strip() because all tokens are read with extra '\n', we need to remove the '\n' character
    #To find index of token, use list.index(token), and to find token at given index, use list[index]
    vocab_hi = [str.strip(text) for text in vocab_hi]

    train_set = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      tf_transform_output,
      batch_size = _batch_size
    )

    eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      tf_transform_output,
      batch_size = _batch_size
    )

    transformer = build_model()

    run_training(epochs = NUM_EPOCHS, train_set, transformer)

    signatures = {
      'serving_default':
        _get_serve_tf_examples_fn(model,tf_transform_output).get_concrete_function(
            tf.TensorSpec(
                shape = [None],
                dtype = tf.string,
                name = 'examples'
            )
        )
    }

    transformer.save(fn_args.serving_model_dir, save_format = 'tf', signatures = signatures)
