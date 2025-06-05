import tensorflow as tf
import time
from seq2seq_model import Encoder, Decoder, evaluate_sample_bpe
import sentencepiece as spm
from gensim.models.fasttext import load_facebook_model
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


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

ft_latin = load_facebook_model("C:/Users/thoma/Desktop/stage/TradNLPlateng/cc.la.300.bin")
ft_english = load_facebook_model("C:/Users/thoma/Desktop/stage/TradNLPlateng/cc.en.300.bin")

# === Paramètres ===
MAX_LEN = 50
BATCH_SIZE = 32
EMBEDDING_DIM = 300
UNITS = 256
EPOCHS = 10
PAD_ID = 0

# === Tokenizer BPE ===
sp = spm.SentencePieceProcessor()
sp.load("bpe_model.model")

# === Dataset TensorFlow ===
dataset = tf.data.experimental.load("C:/Users/thoma/Desktop/stage/TradNLPlateng/processed/tf_dataset_bpe")
dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# === Modèle ===
vocab_size = sp.get_piece_size()


# Préentrainement
embedding_matrix_latin = np.random.uniform(-0.05, 0.05, (vocab_size, EMBEDDING_DIM))
embedding_matrix_english = np.random.uniform(-0.05, 0.05, (vocab_size, EMBEDDING_DIM))

for i in range(vocab_size):
    token = sp.id_to_piece(i)
    if token in ft_latin.wv:
        embedding_matrix_latin[i] = ft_latin.wv[token]
    if token in ft_english.wv:
        embedding_matrix_english[i] = ft_english.wv[token]

encoder = Encoder(vocab_size, EMBEDDING_DIM, UNITS, embedding_matrix_latin)
decoder = Decoder(vocab_size, EMBEDDING_DIM, UNITS, embedding_matrix_english)
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

smoothie = SmoothingFunction().method4

references = []
predictions = []

for batch in dataset.take(20):  # prendre 20 batchs pour test BLEU
    enc_input = batch["encoder_input"]
    dec_target = batch["decoder_target"]

    for i in range(enc_input.shape[0]):  # batch size
        input_sentence = sp.decode(enc_input[i].numpy().tolist())
        target_sentence = sp.decode(dec_target[i].numpy().tolist())

        # Évaluation du modèle sur 1 phrase
        predicted_ids = []
        enc_out, h, c = encoder(tf.expand_dims(enc_input[i], 0))
        dec_input = tf.expand_dims([sp.bos_id()], 0)

        for _ in range(MAX_LEN):
            preds, h, c, _ = decoder(dec_input, enc_out, h, c)
            pred_id = tf.argmax(preds[0]).numpy()
            if pred_id == sp.eos_id():
                break
            predicted_ids.append(pred_id)
            dec_input = tf.expand_dims([pred_id], 0)

        pred_sentence = sp.decode(predicted_ids)

        references.append([target_sentence.split()])
        predictions.append(pred_sentence.split())

# Calcul du score BLEU global
bleu_score = nltk.translate.bleu_score.corpus_bleu(references, predictions, smoothing_function=smoothie)
print(f"\nScore BLEU global : {bleu_score:.4f}")


