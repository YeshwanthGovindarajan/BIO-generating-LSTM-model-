import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

# Load and prepare the dataset
df = pd.read_csv("C:\\Users\\YESHWANTH\\Highperformr\\BIOS\\combined.csv")
df = df.dropna(subset=['BIO'])
bios = df['BIO']
print(bios)

# Tokenizing text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(bios)
tokenized_bios = tokenizer.texts_to_sequences(bios)

# Sequence padding
max_len = max([len(bio) for bio in tokenized_bios])
padded_bios = pad_sequences(tokenized_bios, maxlen=max_len, padding='post')

# Prepare data for training
vocab_size = len(tokenizer.word_index) + 1
X, Y = [], []

for bio in tokenized_bios:
    for i in range(1, len(bio)):
        X.append(bio[:i])
        Y.append(bio[i])

X = pad_sequences(X, maxlen=max_len, padding='post')
Y = to_categorical(Y, num_classes=vocab_size)

# Model architecture
model = Sequential([
    Embedding(vocab_size, 256, input_length=max_len),
    LSTM(512, return_sequences=True),
    LSTM(512),
    Dense(vocab_size, activation='softmax')
])

# Compile model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Callbacks
checkpoint = ModelCheckpoint("best_model.h5", monitor='loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1)

# Fit model
model.fit(X, Y, batch_size=128, epochs=50, callbacks=[checkpoint, early_stopping])

# Function to generate a Twitter bio
def generate_twitter_bio(model, tokenizer, seed_text, max_len, length=20):
    generated_text = seed_text

    for _ in range(length):
        tokenized_seed = tokenizer.texts_to_sequences([generated_text])[0]
        tokenized_seed = pad_sequences([tokenized_seed], maxlen=max_len, padding='post')
        predicted_token = np.argmax(model.predict(tokenized_seed), axis=-1)

        predicted_word = ""
        for word, token in tokenizer.word_index.items():
            if token == predicted_token:
                predicted_word = word
                break
        generated_text += " " + predicted_word

    return generated_text

# Generate and print a bio
seed_text = "Passionate about technology and"
generated_bio = generate_twitter_bio(model, tokenizer, seed_text, max_len)
print("Generated Twitter Bio:", generated_bio)

# Save the tokenizer for future use
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as file:
    file.write(tokenizer_json)

# Load the model and tokenizer for deployment or further use
model = load_model("best_model.h5")
with open('tokenizer.json') as file:
    loaded_tokenizer = Tokenizer.from_json(file.read())

# Evaluate the effectiveness (This could be further expanded with actual metrics and validation sets)
print("Model and tokenizer reloaded successfully and ready for further use or deployment.")
