# ==========================================================
# EXPERIMENT NO: 4
# AIM:
# To implement an information retrieval system using
# TF-IDF and LSA techniques and retrieve relevant
# documents based on a user query.
# ==========================================================

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# -----------------------------
# Step 1: Accept Documents
# -----------------------------
docs = []

n = int(input("Enter number of documents: "))

for i in range(n):
    doc = input(f"Enter document {i+1}: ")
    docs.append(doc)

# -----------------------------
# Step 2: Accept Search Query
# -----------------------------
query = input("\nEnter search query: ")

# -----------------------------
# Step 3: TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

query_vec = vectorizer.transform([query])

# -----------------------------
# Step 4: TF-IDF Similarity
# -----------------------------
tfidf_scores = cosine_similarity(query_vec, X)

print("\n===================================")
print("TF-IDF SIMILARITY SCORES")
print("===================================")

for i, score in enumerate(tfidf_scores[0]):
    print(f"Document {i+1}: {score:.3f}")

# -----------------------------
# Step 5: LSA using SVD
# -----------------------------
# n_components must be less than number of features
components = min(2, X.shape[1] - 1)

if components < 1:
    components = 1

svd = TruncatedSVD(n_components=components, random_state=42)

X_lsa = svd.fit_transform(X)
query_lsa = svd.transform(query_vec)

lsa_scores = cosine_similarity(query_lsa, X_lsa)

print("\n===================================")
print("LSA SIMILARITY SCORES")
print("===================================")

for i, score in enumerate(lsa_scores[0]):
    print(f"Document {i+1}: {score:.3f}")

# -----------------------------
# Step 6: Rank Documents
# -----------------------------
print("\n===================================")
print("RANKING USING TF-IDF")
print("===================================")

tfidf_rank = np.argsort(tfidf_scores[0])[::-1]

for rank, index in enumerate(tfidf_rank, 1):
    print(f"{rank}. Document {index+1} (Score: {tfidf_scores[0][index]:.3f})")

print("\n===================================")
print("RANKING USING LSA")
print("===================================")

lsa_rank = np.argsort(lsa_scores[0])[::-1]

for rank, index in enumerate(lsa_rank, 1):
    print(f"{rank}. Document {index+1} (Score: {lsa_scores[0][index]:.3f})")

# -----------------------------
# Step 7: Most Relevant Document
# -----------------------------
best = np.argmax(lsa_scores[0])

print("\n===================================")
print("MOST RELEVANT DOCUMENT")
print("===================================")

print(docs[best])

print("\n===================================")
print("EXPERIMENT COMPLETED")
print("===================================")