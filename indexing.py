import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import preprocess_text

print("Membaca dataset...")

df = pd.read_csv("dataset/goodreads_books.csv")

print("Jumlah data awal:", len(df))

# Pilih kolom yang digunakan
columns_needed = [
    "bookId",
    "title",
    "author",
    "genres",
    "description",
    "avg_rating"
]

df = df[columns_needed]

# Hapus data kosong
df = df.dropna(subset=[
    "title",
    "author",
    "genres",
    "description"
])

print("Jumlah data setelah cleaning:", len(df))

# Gabungkan semua informasi menjadi satu dokumen
df["document"] = (
    df["title"].astype(str) + " " +
    df["author"].astype(str) + " " +
    df["genres"].astype(str) + " " +
    df["description"].astype(str)
)

print("Melakukan preprocessing...")

df["processed"] = df["document"].apply(preprocess_text)

print("Membuat TF-IDF Matrix...")

vectorizer = TfidfVectorizer(
    max_features=10000
)

tfidf_matrix = vectorizer.fit_transform(
    df["processed"]
)

print("Menyimpan model...")

joblib.dump(
    vectorizer,
    "models/vectorizer.pkl"
)

joblib.dump(
    tfidf_matrix,
    "models/tfidf_matrix.pkl"
)

joblib.dump(
    df,
    "models/books.pkl"
)

print("\nIndexing selesai!")
print("Jumlah dokumen:", len(df))
print("Jumlah fitur:", tfidf_matrix.shape[1])