import tensorflow as tf

# === Bahdanau Attention ===
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, encoder_outputs, hidden_state):
        hidden_state = tf.expand_dims(hidden_state, 1)
        score = self.V(tf.nn.tanh(self.W1(encoder_outputs) + self.W2(hidden_state)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * encoder_outputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, tf.squeeze(attention_weights, -1)

# === Encodeur ===
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, embedding_matrix):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            trainable=True  # ou False si tu veux figer au début
        )
        self.lstm = tf.keras.layers.LSTM(enc_units, return_sequences=True, return_state=True)

    def call(self, x):
        x = self.embedding(x)
        output, h, c = self.lstm(x)
        return output, h, c

# === Décodeur avec Bahdanau Attention ===
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, embedding_matrix):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            trainable=True  # ou False si tu veux figer au début
        )
        self.lstm = tf.keras.layers.LSTM(dec_units, return_sequences=True, return_state=True)
        self.attention = BahdanauAttention(dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, enc_output, hidden, cell):
        # 1. Appliquer l'attention
        context_vector, attention_weights = self.attention(enc_output, hidden)

        # 2. Embedding du token courant
        x = self.embedding(x)  # (batch_size, 1, embedding_dim)

        # 3. Concaténation context + embedding
        context_vector = tf.expand_dims(context_vector, 1)  # (batch_size, 1, dec_units)
        x = tf.concat([context_vector, x], axis=-1)  # (batch_size, 1, embedding_dim + dec_units)

        # 4. Passer dans le LSTM avec les états précédents
        output, state_h, state_c = self.lstm(x, initial_state=[hidden, cell])

        # 5. Prédiction finale
        output = tf.reshape(output, (-1, output.shape[2]))  # (batch_size, dec_units)
        logits = self.fc(output)  # (batch_size, vocab_size)

        return logits, state_h, state_c, attention_weights




def evaluate_sample_bpe(sentence, encoder, decoder, sp, units=128, max_length=30):
    import numpy as np
    import tensorflow as tf

    # Encodage BPE de la phrase source
    sentence_bpe = sp.encode(sentence, out_type=int)
    inputs = tf.convert_to_tensor([sentence_bpe])

    # Masques initiaux
    encoder_output, hidden1, hidden2 = encoder(inputs)
    dec_input = tf.expand_dims([sp.bos_id()], 0)

    result = []

    for t in range(max_length):
        predictions, hidden1, hidden2, _= decoder(dec_input, encoder_output, hidden1, hidden2)

        predicted_id = tf.argmax(tf.squeeze(predictions, axis=0), axis=-1).numpy()


        if predicted_id == sp.eos_id():
            break

        result.append(predicted_id)
        dec_input = tf.expand_dims([predicted_id], 0)

    print("Traduction complète :", sp.decode([int(i) for i in result]))



