import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft

TRANSFORMED_TRAIN_DATA_FILEBASE = 'train_transformed'

_lang1 = 'lang1'
_lang2 = 'lang2'

_vocab_size = 100000
_max = 64

def _transformed_name(key, is_input = False):
  if (is_input == True):
    return key + '_xf_input'

  return key+'_xf'

def return_tokens(text):
  return tf.strings.split(tf.reshape(text,[-1])).to_sparse()

def vectorized(tokenize):

  convert_to_dense = tf.sparse.to_dense(tokenize, default_value = -1)

  padding_config = [[0,0], [0, _max]]
  dense_padded = tf.pad(convert_to_dense, padding_config, 'CONSTANT', -1)
  dense_max_len = tf.slice(dense_padded, [0,0], [-1, _max])
  dense_max_len += 1


  return dense_max_len

def preprocessing_fn(inputs):

  eng = return_tokens(inputs[_en])
  hin = return_tokens(inputs[_hi])

  tokenize_en = tft.compute_and_apply_vocabulary(
      eng, default_value = _vocab_size, top_k = _vocab_size, vocab_filename = 'vocab_en' 
  )

  tokenize_hi = tft.compute_and_apply_vocabulary(
      hin, default_value = _vocab_size, top_k = _vocab_size, vocab_filename = 'vocab_hi'
  )

  data = {
      _transformed_name(_en, True): vectorized(tokenize_en),
      _transformed_name(_hi): vectorized(tokenize_hi),
  }

  return data
