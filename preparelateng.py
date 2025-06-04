import os
import tensorflow as tf
import sentencepiece as spm
import numpy as np

MAX_LEN =100

# === 1. Chargement brut ===
latin_sents = []
en_sents = []

with open("C:/Users/thoma/Desktop/stage/TradNLPlateng/latin_en_cgrosenthal.csv", encoding="utf-8") as f:
    for line in f:
        if ";" in line:
            parts = line.strip().split(";")
            if len(parts) >= 2:
                la, en = parts[0].strip(), parts[1].strip()
                en_sents.append(en)
                latin_sents.append(la)

print(f"{len(latin_sents)} paires chargées.")

# === 2. Préparer corpus BPE pour entraînement SentencePiece ===
with open("C:/Users/thoma/Desktop/stage/TradNLPlateng/bpe_corpus.txt", "w", encoding="utf-8") as f:
    for s in latin_sents + en_sents:
        f.write(s + "\n")

# === 3. Entraîner SentencePiece BPE ===
spm.SentencePieceTrainer.Train(
    input='C:/Users/thoma/Desktop/stage/TradNLPlateng/bpe_corpus.txt',
    model_prefix='C:/Users/thoma/Desktop/stage/TradNLPlateng/bpe_model',
    vocab_size=32000,
    model_type='bpe',
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)

# === 4. Charger le modèle BPE ===
sp = spm.SentencePieceProcessor()
sp.load("C:/Users/thoma/Desktop/stage/TradNLPlateng/bpe_model.model")

# === 5. Encodage des phrases en ID ===
src_seqs = []
decoder_inputs = []
decoder_targets = []
skipped = 0

for lat, en in zip(latin_sents, en_sents):
    lat_ids = sp.encode(lat, out_type=int)
    en_ids = sp.encode(en, out_type=int)
    en_ids.append(3)  # Ajout du token <eos>

    if len(lat_ids) <= MAX_LEN and len(en_ids) <= MAX_LEN:
        lat_padded = lat_ids + [0] * (MAX_LEN - len(lat_ids))
        dec_input = en_ids[:-1] + [0] * (MAX_LEN - len(en_ids[:-1]))
        dec_target = en_ids[1:] + [0] * (MAX_LEN - len(en_ids[1:]))

        if len(src_seqs) == 0:
            print(" Exemple:")
            print("  Latin (tokens):", sp.decode_ids(lat_ids))
            print("  Anglais (tokens):", sp.decode_ids(en_ids))
            print("  Decoder target:", dec_target)

        src_seqs.append(lat_padded)
        decoder_inputs.append(dec_input)
        decoder_targets.append(dec_target)
    else:
        skipped += 1

print(f" {len(src_seqs)} paires encodées.")
print(f" {skipped} paires ignorées car > {MAX_LEN} tokens.")

# === 6. Construction du dataset TensorFlow ===
dataset = tf.data.Dataset.from_tensor_slices({
    "encoder_input": tf.constant(src_seqs, dtype=tf.int32),
    "decoder_input": tf.constant(decoder_inputs, dtype=tf.int32),
    "decoder_target": tf.constant(decoder_targets, dtype=tf.int32),
})

os.makedirs("C:/Users/thoma/Desktop/stage/TradNLPlateng/processed/tf_dataset_bpe", exist_ok=True)
tf.data.experimental.save(dataset, "C:/Users/thoma/Desktop/stage/TradNLPlateng/processed/tf_dataset_bpe")

print(" Dataset BPE sauvegardé avec succès.")

