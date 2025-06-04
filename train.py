import tensorflow as tf
import time
from seq2seq_model import Encoder, Decoder, evaluate_sample_bpe
import sentencepiece as spm

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU activé :", gpus[0])
    except RuntimeError as e:
        print("Erreur GPU :", e)
else:
    print("Aucun GPU détecté")

# === Paramètres ===
MAX_LEN = 100
BATCH_SIZE = 32
EMBEDDING_DIM = 256
UNITS = 256
EPOCHS = 5
PAD_ID = 0

# === Tokenizer BPE ===
sp = spm.SentencePieceProcessor()
sp.load("bpe_model.model")

# === Dataset TensorFlow ===
dataset = tf.data.experimental.load("C:/Users/thoma/Desktop/stage/TradNLPlateng/processed/tf_dataset_bpe")
dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# === Modèle ===
vocab_size = sp.get_piece_size()
encoder = Encoder(vocab_size, EMBEDDING_DIM, UNITS)
decoder = Decoder(vocab_size, EMBEDDING_DIM, UNITS)
optimizer = tf.keras.optimizers.Adam()

# Nouvelle loss function
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')

def smooth_one_hot(y_true, vocab_size, smoothing=0.1):
    conf = 1.0 - smoothing
    low_conf = smoothing / (vocab_size - 1)
    one_hot = tf.one_hot(y_true, depth=vocab_size)
    return one_hot * conf + (1 - one_hot) * low_conf


loss_history = []

@tf.function
def train_step(batch):
    loss = 0.0
    enc_input = batch["encoder_input"]
    dec_input = batch["decoder_input"]
    dec_target = batch["decoder_target"]

    with tf.GradientTape() as tape:
        enc_output, enc_h, enc_c = encoder(enc_input)
        dec_h, dec_c = enc_h, enc_c

        for t in range(MAX_LEN):
            x_t = tf.expand_dims(dec_input[:, t], 1)
            preds, dec_h, dec_c, _ = decoder(x_t, enc_output, dec_h, dec_c)
            y_t = dec_target[:, t]
            mask = tf.cast(tf.not_equal(y_t, PAD_ID), tf.float32)
            smoothed_y = smooth_one_hot(y_t, vocab_size, smoothing=0.1)
            loss_step = loss_fn(smoothed_y, preds)

            loss += tf.reduce_mean(loss_step * mask)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss / MAX_LEN

# === Entraînement ===
for epoch in range(EPOCHS):
    start = time.time()
    total_loss = 0.0
    interval_loss = 0.0
    interval_steps = 0
    loss_history.clear()

    for step, batch in enumerate(dataset):
        batch_loss = train_step(batch)
        total_loss += batch_loss
        interval_loss += batch_loss
        interval_steps += 1
        loss_history.append(float(batch_loss.numpy()))

        if (step + 1) % 100 == 0:
            avg_interval = interval_loss / interval_steps
            tf.print(f" Époque {epoch+1} - Batchs {step-99}-{step} - Perte moyenne :", avg_interval)
            interval_loss = 0.0
            interval_steps = 0

    avg_loss = total_loss / (step + 1)
    tf.print(f" Époque {epoch+1} terminée en", time.time() - start, "sec - Perte moyenne :", avg_loss)

    encoder.save_weights("C:/Users/thoma/Desktop/stage/TradNLPlateng/checkpoints/encoder.h5")
    decoder.save_weights("C:/Users/thoma/Desktop/stage/TradNLPlateng/checkpoints/decoder.h5")
    print("Modèles sauvegardés.")




