import tensorflow as tf
import sentencepiece as spm
import numpy as np
from transformer import build_transformer, MAX_LEN, VOCAB_SIZE

# Charger le tokenizer
sp = spm.SentencePieceProcessor()
sp.load('C:/Users/Corniere/Desktop/stage/TradNLPlateng/bpe_model.model')

# Paramètres du modèle
num_layers = 2
d_model = 128
num_heads = 4
dff = 512

# Fonction d'encodage
def encode_input(sentence):
    ids = sp.encode(sentence, out_type=int)
    ids = [1] + ids[:MAX_LEN - 2] + [2]
    padded = ids + [0] * (MAX_LEN - len(ids))
    return np.array(padded)

# Fonction de décodage auto-régressive
def predict(model, sentence):
    encoder_input = encode_input(sentence)
    encoder_input = tf.expand_dims(encoder_input, 0)  # batch size = 1

    decoder_input = [1]  # BOS token
    for _ in range(MAX_LEN - 1):
        dec_input = tf.expand_dims(decoder_input + [0]*(MAX_LEN - len(decoder_input)), 0)
        predictions = model([encoder_input, dec_input], training=False)
        next_token_logits = predictions[0, len(decoder_input)-1]
        next_token = tf.argmax(next_token_logits).numpy()

        if next_token == 2:  # EOS
            break
        decoder_input.append(next_token)
    tokens = [int(i) for i in decoder_input[1:]]
    
    print("Tokens générés :", decoder_input)
    print("IDs sans BOS :", tokens)
    print("Tokens texte :", sp.id_to_piece(tokens))

    return sp.decode_ids([int(i) for i in decoder_input[1:]])

  # On enlève le BOS


print(sp.encode("Salve, quid agis?", out_type=str))

# Charger le modèle et les poids
model = build_transformer(num_layers, d_model, num_heads, dff)
model.load_weights('C:/Users/Corniere/checkpoint/transformer_epoch_10.ckpt')  # Mets le bon chemin vers tes poids

# Test de quelques phrases
phrases = [
    "Salve, quid agis?",
    "Feles in sella sedet.",
    "Cras ad forum ibimus.",
    "Olera non amo."
]

for phrase in phrases:
    traduction = predict(model, phrase)
    print(f"> {phrase}\n= {traduction}\n")

