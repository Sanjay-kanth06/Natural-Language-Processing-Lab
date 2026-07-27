# ==========================================================
# EXPERIMENT NO: 5
# AIM:
# To implement Named Entity Recognition (NER) using NLTK
# and identify named entities such as person names,
# organizations, and locations in legal text documents.
# ==========================================================

import nltk
from nltk import word_tokenize, pos_tag

# Download required datasets
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# -----------------------------
# Step 1: Accept Legal Text
# -----------------------------
text = input("Enter legal text:\n")

# -----------------------------
# Step 2: Tokenization
# -----------------------------
tokens = word_tokenize(text)

# -----------------------------
# Step 3: POS Tagging
# -----------------------------
tags = pos_tag(tokens)

# -----------------------------
# Step 4: Detect Named Entities
# -----------------------------
print("\n===================================")
print("DETECTED NAMED ENTITIES")
print("===================================")

count = 0
entities = []

for word, tag in tags:
    if tag == "NNP":
        print(f"{word} -> ENTITY")
        entities.append(word)
        count += 1

# -----------------------------
# Step 5: Display Entity Count
# -----------------------------
print("\nPredicted Entity Count:", count)

# -----------------------------
# Step 6: Accept Actual Count
# -----------------------------
actual = int(input("\nEnter actual number of entities: "))

# -----------------------------
# Step 7: Calculate Accuracy
# -----------------------------
if max(count, actual) == 0:
    accuracy = 100
else:
    accuracy = (min(count, actual) / max(count, actual)) * 100

# -----------------------------
# Step 8: Display Results
# -----------------------------
print("\n===================================")
print("RESULT")
print("===================================")

print("Predicted Entities:", count)
print("Actual Entities   :", actual)
print("NER Accuracy      :", round(accuracy, 2), "%")

print("\nList of Predicted Entities:")
if entities:
    for entity in entities:
        print("-", entity)
else:
    print("No named entities detected.")

print("\n===================================")
print("EXPERIMENT COMPLETED")
print("===================================")