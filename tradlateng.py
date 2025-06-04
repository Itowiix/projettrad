import tensorflow as tf
import sentencepiece as spm
from seq2seq_model import Encoder, Decoder, evaluate_sample_bpe

# === Chemins des modèles sauvegardés ===
CHECKPOINT_DIR = "C:/Users/thoma/Desktop/stage/TradNLPlateng/checkpoints/"
BPE_MODEL_PATH = "bpe_model.model"

# === Paramètres du modèle ===
MAX_LEN = 30
EMBEDDING_DIM = 256
UNITS = 128

# === Chargement du tokenizer SentencePiece ===
sp = spm.SentencePieceProcessor()
sp.load(BPE_MODEL_PATH)
vocab_size = sp.get_piece_size()

# === Instanciation du modèle ===
encoder = Encoder(vocab_size, EMBEDDING_DIM, UNITS)
decoder = Decoder(vocab_size, EMBEDDING_DIM, UNITS)

# === Initialisation des variables du modèle ===
dummy_input = tf.constant([sp.encode("dummy", out_type=int)], dtype=tf.int32)
encoder(dummy_input)  # appel pour créer les variables
decoder(tf.constant([[1]]), tf.random.normal((1, MAX_LEN, UNITS)), 
        tf.zeros((1, UNITS)), tf.zeros((1, UNITS)))  # idem pour le decoder

# === Chargement des poids sauvegardés ===
encoder.load_weights(CHECKPOINT_DIR + "encoder.h5")
decoder.load_weights(CHECKPOINT_DIR + "decoder.h5")

# === Exemple de traduction ===
while True:
    user_input = input("Entrez une phrase latine à traduire (ou 'exit') : ").strip()
    if user_input.lower() == 'exit':
        break
    print(" Traduction :")
    evaluate_sample_bpe(user_input, encoder, decoder, sp)
    print()
