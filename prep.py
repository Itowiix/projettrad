import os
import tensorflow as tf
import sentencepiece as spm
import numpy as np
import csv

MAX_LEN = 50
VOCAB_SIZE = 8000
DATA_DIR = "C:/Users/Corniere/Desktop/stage/TradNLPlateng"
CSV_FILE = f"{DATA_DIR}/latin_en_cgrosenthal.csv"

# === 1. Charger données CSV ===
latin_sents = []
en_sents = []
with open(CSV_FILE, encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=";")
    for row in reader:
        if len(row) >= 2:
            latin_sents.append(row[0].strip())
            en_sents.append(row[1].strip())
print(f"✅ {len(latin_sents)} paires chargées.")

# === 2. Sauver corpus brut pour SentencePiece ===
os.makedirs(f"{DATA_DIR}/corpus", exist_ok=True)
with open(f"{DATA_DIR}/corpus/latin.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(latin_sents))
with open(f"{DATA_DIR}/corpus/english.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(en_sents))

# === 3. Entraîner 2 modèles SentencePiece ===
spm.SentencePieceTrainer.Train(
    input=f"{DATA_DIR}/corpus/latin.txt",
    model_prefix=f"{DATA_DIR}/models/sp_lat",
    vocab_size=VOCAB_SIZE,
    model_type="bpe",
    character_coverage=1.0,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3
)
spm.SentencePieceTrainer.Train(
    input=f"{DATA_DIR}/corpus/english.txt",
    model_prefix=f"{DATA_DIR}/models/sp_en",
    vocab_size=VOCAB_SIZE,
    model_type="bpe",
    character_coverage=1.0,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3
)
print("✅ Tokenizers entraînés.")

# === 4. Charger les modèles SentencePiece ===
sp_lat = spm.SentencePieceProcessor()
sp_lat.load(f"{DATA_DIR}/models/sp_lat.model")
sp_en = spm.SentencePieceProcessor()
sp_en.load(f"{DATA_DIR}/models/sp_en.model")

# === 5. Encodage et padding des séquences ===
src_seqs, dec_inputs, dec_targets = [], [], []
skipped = 0

for la, en in zip(latin_sents, en_sents):
    lat_ids = [sp_lat.bos_id()] + sp_lat.encode(la, out_type=int) + [sp_lat.eos_id()]
    en_ids = [sp_en.bos_id()] + sp_en.encode(en, out_type=int) + [sp_en.eos_id()]

    if len(lat_ids) <= MAX_LEN and len(en_ids) <= MAX_LEN:
        lat_pad = lat_ids + [sp_lat.pad_id()] * (MAX_LEN - len(lat_ids))
        dec_in = en_ids[:-1] + [sp_en.pad_id()] * (MAX_LEN - len(en_ids[:-1]))
        dec_tg = en_ids[1:] + [sp_en.pad_id()] * (MAX_LEN - len(en_ids[1:]))

        src_seqs.append(lat_pad)
        dec_inputs.append(dec_in)
        dec_targets.append(dec_tg)
    else:
        skipped += 1

print(f"✅ {len(src_seqs)} paires encodées, {skipped} ignorées (> {MAX_LEN}).")

# === 6. Construire le tf.data.Dataset ===
dataset = tf.data.Dataset.from_tensor_slices({
    "encoder_input": tf.constant(src_seqs, dtype=tf.int32),
    "decoder_input": tf.constant(dec_inputs, dtype=tf.int32),
    "decoder_target": tf.constant(dec_targets, dtype=tf.int32),
})

out_path = f"{DATA_DIR}/processed/tf_dataset_dual"
os.makedirs(out_path, exist_ok=True)
tf.data.experimental.save(dataset, out_path)

print("✅ Dataset TensorFlow sauvegardé dans :", out_path)
