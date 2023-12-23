# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import tensorflow as tf
from extact_text_pdf import extract_text_pdf
from tensorflow.keras.layers import LSTM, Dense, Embedding

# extract the text from the pdf

book_text = extract_text_pdf(r"C:\Users\Oukhm\OneDrive\Bureau\Rabit Hole\AI\weak-to-strong-generalization.pdf")

tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts([book_text])

# Convert text to a sequence of integers
seq = tokenizer.texts_to_sequences([book_text])[0]

# Prepare dataset - create input and output pairs
seq_length = 40
inputs, targets = [], []
for i in range(0, len(seq) - seq_length):
    inputs.append(seq[i:i + seq_length])
    targets.append(seq[i + seq_length])

# Pad sequences
inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs)

# Build LSTM model

vocab_size = len(tokenizer.word_index) + 1

model = tf.keras.sequential([
    Embedding(vocab_size, 50, input_length=seq_length),
    LSTM(100, return_sequences=True),
    LSTM(100),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Convert targets to numpy array
targets = np.array(targets)

# Train the model
model.fit(inputs, targets, epochs=10, batch_size=128)

model.save('my_language_model.h5')


# Function to generate text
def generate_text(seed_text, next_chars=100):
    for _ in range(next_chars):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tf.keras.pad_sequences([token_list], maxlen=seq_length, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_char = ''
        for char, index in tokenizer.word_index.items():
            if index == predicted:
                output_char = char
                break
        seed_text += output_char
    return seed_text


print(generate_text("This is a beginning of a sentence,"))


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    print(tf.__version__)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
