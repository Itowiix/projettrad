import tensorflow as tf
import pandas as pd
import sentencepiece as spm
import numpy as np

# ========== 1. Chargement du tokenizer ==========
sp = spm.SentencePieceProcessor()
sp.load('C:/Users/Corniere/Desktop/stage/TradNLPlateng/bpe_model.model')
VOCAB_SIZE = sp.get_piece_size()
MAX_LEN = 50

# ========== 2. Préparer les données ==========
def encode_sentence(text):
    ids = sp.encode(text.numpy().decode('utf-8'), out_type=int)
    ids = ids[:MAX_LEN - 2]
    return [2] + ids + [3]  # BOS=1, EOS=2

def tf_encode_sentence(text):
    return tf.py_function(func=encode_sentence, inp=[text], Tout=tf.int32)


# ========== 4. Masquage ==========
def create_padding_mask(seq):
    return tf.cast(tf.math.equal(seq, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]  # (batch, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


# ========== 5. Architecture Transformer complète ==========
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, pos, i, d_model):
        pos = tf.cast(pos, tf.float32)
        i = tf.cast(i, tf.float32)
        d_model = tf.cast(d_model, tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / d_model)
        return pos * angle_rates


    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(tf.range(position)[:, tf.newaxis],
                                     tf.range(d_model)[tf.newaxis, :],
                                     d_model)
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        return tf.cast(tf.concat([sines, cosines], axis=-1), dtype=tf.float32)[tf.newaxis, ...]

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_output = self.att(x, x, x)
        x = self.norm1(x + self.dropout(attn_output, training=training))
        ffn_output = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_output, training=training))


class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.self_attn = tf.keras.layers.MultiHeadAttention(num_heads, key_dim=d_model)
        self.cross_attn = tf.keras.layers.MultiHeadAttention(num_heads, key_dim=d_model)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
        ])

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

    def call(self, x, enc_output, training=None, look_ahead_mask=None, padding_mask=None):
        # Masked self-attention
        attn1 = self.self_attn(x, x, x, attention_mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.norm1(x + attn1)

        # Cross-attention
        attn2 = self.cross_attn(out1, enc_output, enc_output, attention_mask=padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.norm2(out1 + attn2)

        # Feed-forward
        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        return self.norm3(out2 + ffn_out)




def build_transformer(num_layers, d_model, num_heads, dff):
    enc_inputs = tf.keras.Input(shape=(MAX_LEN,), name='encoder_inputs')
    dec_inputs = tf.keras.Input(shape=(MAX_LEN,), name='decoder_inputs')

    # Masques
    enc_padding_mask = tf.keras.layers.Lambda(create_padding_mask)(enc_inputs)
    look_ahead_mask = tf.keras.layers.Lambda(
        lambda x: create_look_ahead_mask(MAX_LEN)
    )(dec_inputs)
    dec_padding_mask = enc_padding_mask  # même masque

    # Encodeur
    x_enc = tf.keras.layers.Embedding(VOCAB_SIZE, d_model)(enc_inputs)
    x_enc = PositionalEncoding(1000, d_model)(x_enc)
    for _ in range(num_layers):
        x_enc = TransformerEncoderLayer(d_model, num_heads, dff)(x_enc)

    # Décodeur
    x_dec = tf.keras.layers.Embedding(VOCAB_SIZE, d_model)(dec_inputs)
    x_dec = PositionalEncoding(1000, d_model)(x_dec)
    for _ in range(num_layers):
        x_dec = TransformerDecoderLayer(d_model, num_heads, dff)(
            x_dec, x_enc,
            training=True,  # ← CET ARGUMENT MANQUAIT
            look_ahead_mask=look_ahead_mask,
            padding_mask=dec_padding_mask
        )


    outputs = tf.keras.layers.Dense(VOCAB_SIZE)(x_dec)
    return tf.keras.Model(inputs=[enc_inputs, dec_inputs], outputs=outputs)


# ========== 6. Compilation et Entraînement ==========
model = build_transformer(num_layers=2, d_model=256, num_heads=8, dff=1024)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def masked_loss(y_true, y_pred ):
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = loss_fn(y_true, y_pred)
    return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

def rename_keys(example):
    return {
        "encoder_inputs": example["encoder_input"],
        "decoder_inputs": example["decoder_input"]
    }, example["decoder_target"]


checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "checkpoint/transformer_epoch_{epoch}.ckpt",
    save_weights_only=True,
    save_best_only=False,
    verbose=1
)

element_spec = {
    "encoder_input": tf.TensorSpec(shape=(50,), dtype=tf.int32),
    "decoder_input": tf.TensorSpec(shape=(50,), dtype=tf.int32),
    "decoder_target": tf.TensorSpec(shape=(50,), dtype=tf.int32),
}

dataset = tf.data.experimental.load(
    "C:/Users/Corniere/Desktop/stage/TradNLPlateng/processed2/tf_dataset_bpe",
    element_spec=element_spec
)

dataset = dataset.map(rename_keys)
# Optionnel : le batch n’est pas enregistré, donc on le refait
dataset = dataset.shuffle(5000).batch(32).prefetch(tf.data.AUTOTUNE)


model.compile(optimizer='adam', loss=masked_loss, metrics=['accuracy'])
for _, target in dataset.take(10):
    y = target.numpy()
    if np.max(y) >= VOCAB_SIZE:
        print("❌ ERREUR: ID hors vocabulaire :", np.max(y))

if __name__ == '__main__':
    model.fit(dataset, epochs=10, callbacks=[checkpoint_cb])