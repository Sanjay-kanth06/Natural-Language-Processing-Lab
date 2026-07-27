import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score

# Download required tokenizer
nltk.download('punkt')

# Relation keywords
keywords = ["treats", "reduces", "controls", "helps"]

# Accept input
sentence = input("Enter biomedical sentence: ")
actual = int(input("Actual Relation (1/0): "))

# Convert to lowercase
sentence = sentence.lower()

# Tokenize
tokens = word_tokenize(sentence)

print("\nTokens:")
print(tokens)

# Predict relation using rule-based approach
predicted = 0
for word in tokens:
    if word in keywords:
        predicted = 1
        break

print("\nPredicted Relation:", predicted)

# Evaluation
y_true = [actual]
y_pred = [predicted]

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\nEvaluation Results")
print("------------------")
print("Precision :", precision)
print("Recall    :", recall)
print("F1-Score  :", f1)