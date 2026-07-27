# ==========================================================
# EXPERIMENT NO: 3
# AIM:
# To calculate text similarity using TF-IDF and WordNet
# similarity and group similar news headlines into topics.
# ==========================================================

import nltk
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Download WordNet data
nltk.download('wordnet')
nltk.download('omw-1.4')

# -----------------------------
# Step 1: Accept Headlines
# -----------------------------
headlines = []

n = int(input("Enter number of headlines: "))

for i in range(n):
    headline = input(f"Enter headline {i+1}: ")
    headlines.append(headline)

# -----------------------------
# Step 2: Convert to TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(headlines)

# -----------------------------
# Step 3: Cosine Similarity
# -----------------------------
print("\n==============================")
print("COSINE SIMILARITY MATRIX")
print("==============================")

similarity_matrix = cosine_similarity(X)

for row in similarity_matrix:
    print(["{:.2f}".format(value) for value in row])

# -----------------------------
# Step 4: K-Means Clustering
# -----------------------------
clusters = min(2, len(headlines))

kmeans = KMeans(
    n_clusters=clusters,
    random_state=0,
    n_init=10
)

kmeans.fit(X)

print("\n==============================")
print("HEADLINE CLUSTERS")
print("==============================")

for i in range(len(headlines)):
    print(f"{headlines[i]} --> Cluster {kmeans.labels_[i]}")

# -----------------------------
# Step 5: WordNet Similarity
# -----------------------------
print("\n==============================")
print("WORDNET SIMILARITY")
print("==============================")

w1 = input("Enter first word: ")
w2 = input("Enter second word: ")

synsets1 = wn.synsets(w1)
synsets2 = wn.synsets(w2)

if synsets1 and synsets2:
    similarity = synsets1[0].path_similarity(synsets2[0])

    if similarity is not None:
        print(f"\nWordNet Similarity between '{w1}' and '{w2}' = {similarity:.3f}")
    else:
        print("\nNo similarity could be computed.")
else:
    print("\nOne or both words are not found in WordNet.")

print("\n==============================")
print("EXPERIMENT COMPLETED")
print("==============================")