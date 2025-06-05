import tensorflow as tf
import time
from seq2seq_model import Encoder, Decoder, evaluate_sample_bpe
import sentencepiece as spm
from gensim.models.fasttext import load_facebook_model
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt

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

print("check")
# === Paramètres ===
MAX_LEN = 50
BATCH_SIZE = 32
EMBEDDING_DIM = 300
UNITS = 256
EPOCHS = 5
PAD_ID = 0
epoch_losses = []
epoch_bleus = []

print("check")

def get_teacher_forcing_ratio(epoch, total_epochs):
    # Exemple linéaire : de 1.0 à 0.3
    start_ratio = 1.0
    end_ratio = 0.3
    return max(end_ratio, start_ratio - ((start_ratio - end_ratio) * (epoch / total_epochs)))

# === Tokenizer BPE ===
sp_lat = spm.SentencePieceProcessor()
sp_lat.load("C:/Users/Corniere/Desktop/stage/TradNLPlateng/models/sp_lat.model")

sp_en = spm.SentencePieceProcessor()
sp_en.load("C:/Users/Corniere/Desktop/stage/TradNLPlateng/models/sp_en.model")


# === Dataset TensorFlow ===
dataset = tf.data.experimental.load("C:/Users/Corniere/Desktop/stage/TradNLPlateng/processed/tf_dataset_dual")
dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
valid_dataset = tf.data.experimental.load("C:/Users/Corniere/Desktop/stage/TradNLPlateng/processed/tf_dataset_dual_valid")
valid_dataset = valid_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
sample = next(iter(valid_dataset))

dec_input = sample["decoder_input"][0].numpy()
dec_target = sample["decoder_target"][0].numpy()

print("decoder_input :", dec_input[:10])
print("decoder_target:", dec_target[:10])

print("check")

# === Modèle ===
vocab_size_lat = sp_lat.get_piece_size()
vocab_size_en = sp_en.get_piece_size()



embedding_matrix_latin = np.load("C:/Users/Corniere/Desktop/stage/TradNLPlateng/embedding_matrix_latin.npy")
embedding_matrix_english = np.load("C:/Users/Corniere/Desktop/stage/TradNLPlateng/embedding_matrix_english.npy")

encoder = Encoder(vocab_size_lat, EMBEDDING_DIM, UNITS, embedding_matrix_latin)
decoder = Decoder(vocab_size_en, EMBEDDING_DIM, UNITS, embedding_matrix_english)

optimizer = tf.keras.optimizers.Adam()
print("check")

# Nouvelle loss function
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')

def smooth_one_hot(y_true, vocab_size, smoothing=0.1):
    conf = 1.0 - smoothing
    low_conf = smoothing / (vocab_size - 1)
    one_hot = tf.one_hot(y_true, depth=vocab_size)
    return one_hot * conf + (1 - one_hot) * low_conf


loss_history = []

@tf.function
def train_step(batch, teacher_forcing_ratio=1.0):
    loss = 0.0
    enc_input = batch["encoder_input"]
    dec_input = batch["decoder_input"]
    dec_target = batch["decoder_target"]

    batch_size = tf.shape(enc_input)[0]
    enc_output, enc_h, enc_c = encoder(enc_input)
    dec_h, dec_c = enc_h, enc_c

    # Début du décodage : BOS
    dec_input_t = tf.expand_dims(dec_input[:, 0], 1)

    for t in range(1, MAX_LEN):  # On saute t=0 car déjà utilisé
        preds, dec_h, dec_c, _ = decoder(dec_input_t, enc_output, dec_h, dec_c)
        y_t = dec_target[:, t]

        # Smooth one-hot target
        smoothed_y = smooth_one_hot(y_t, vocab_size_en, smoothing=0.1)

        # Masking
        mask = tf.cast(tf.not_equal(y_t, PAD_ID), tf.float32)
        loss_step = loss_fn(smoothed_y, preds)
        loss += tf.reduce_mean(loss_step * mask)

        # Teacher forcing ou pas ?
        predicted_id = tf.argmax(preds, axis=1, output_type=tf.int32)
        use_teacher = tf.random.uniform([]) < teacher_forcing_ratio
        next_input = tf.where(use_teacher, dec_input[:, t], predicted_id)

        # Préparer entrée suivante
        dec_input_t = tf.expand_dims(next_input, 1)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tf.gradients(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss / tf.cast(MAX_LEN, tf.float32)
print("check")


def compute_bleu_score(dataset, encoder, decoder, sp_lat, sp_en):
    smooth = SmoothingFunction().method1
    scores = []
    for batch in dataset.take(100):  # max 100 batches = rapide
        enc_input = batch["encoder_input"]
        dec_target = batch["decoder_target"]

        enc_output, enc_h, enc_c = encoder(enc_input)
        dec_h, dec_c = enc_h, enc_c

        dec_input_t = tf.expand_dims([sp_en.bos_id()] * enc_input.shape[0], 1)
        preds = []

        for t in range(MAX_LEN):
            logits, dec_h, dec_c, _ = decoder(dec_input_t, enc_output, dec_h, dec_c)
            predicted_ids = tf.argmax(logits, axis=1, output_type=tf.int32)
            preds.append(predicted_ids)
            dec_input_t = tf.expand_dims(predicted_ids, 1)

        preds = tf.stack(preds, axis=1).numpy()
        targets = dec_target.numpy()

        for pred, ref in zip(preds, targets):
            try:
                # enlever les PAD et EOS
                pred_tokens = [tok for tok in pred if tok != PAD_ID and tok != sp_en.eos_id()]
                ref_tokens = [tok for tok in ref if tok != PAD_ID and tok != sp_en.eos_id()]
                if len(ref_tokens) == 0 or len(pred_tokens) == 0:
                    continue
                score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
                scores.append(score)
            except:
                continue
    return np.mean(scores)

# === Entraînement ===
for epoch in range(EPOCHS):
    teacher_ratio = get_teacher_forcing_ratio(epoch, EPOCHS)
    start = time.time()
    total_loss = 0.0
    interval_loss = 0.0
    interval_steps = 0
    loss_history.clear()

    for step, batch in enumerate(dataset):
        batch_loss = train_step(batch, teacher_forcing_ratio=teacher_ratio)
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
    sample_lat = "Gallia est omnis divisa in partes tres."
    evaluate_sample_bpe(sample_lat, encoder, decoder, sp_lat, 256, 50)   
    epoch_losses.append(float(avg_loss.numpy()))
    bleu_score = compute_bleu_score(valid_dataset, encoder, decoder, sp_lat, sp_en)
    epoch_bleus.append(bleu_score)
    print(f"BLEU score validation : {bleu_score:.4f}")
    encoder.save_weights("C:/Users/Corniere/Desktop/stage/TradNLPlateng/checkpoints/encoder.h5")
    decoder.save_weights("C:/Users/Corniere/Desktop/stage/TradNLPlateng/checkpoints/decoder.h5")
    print("Modèles sauvegardés.")
plt.figure()
plt.plot(range(1, EPOCHS + 1), epoch_losses, label='Loss')
plt.plot(range(1, EPOCHS + 1), epoch_bleus, label='BLEU')
plt.xlabel("Epoch")
plt.ylabel("Valeurs")
plt.title("Courbes de performance")
plt.legend()
plt.grid()
plt.savefig("courbes_performance.png")
plt.show()