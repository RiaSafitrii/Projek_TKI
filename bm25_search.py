import joblib
import numpy as np

from rank_bm25 import BM25Okapi

from preprocessing import preprocess_text

print("Memuat data BM25...")

books = joblib.load(
    "models/books.pkl"
)

corpus = books["processed"].tolist()

tokenized_corpus = [
    doc.split()
    for doc in corpus
]

bm25 = BM25Okapi(tokenized_corpus)

print("BM25 siap digunakan.")


def search_books_bm25(query, top_k=10):

    query = preprocess_text(query)

    tokenized_query = query.split()

    scores = bm25.get_scores(
        tokenized_query
    )

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:

        results.append({

    "bookId":
    books.iloc[idx]["bookId"],

    "title":
    books.iloc[idx]["title"],

    "author":
    books.iloc[idx]["author"],

    "genres":
    books.iloc[idx]["genres"],

    "rating":
    books.iloc[idx]["avg_rating"],

    "description":
    str(books.iloc[idx]["description"])[:250],

    "score":
    round(float(scores[idx]), 4)

})

    return results


if __name__ == "__main__":

    query = input(
        "\nMasukkan Query: "
    )

    results = search_books_bm25(query)

    print("\nHASIL BM25\n")

    for i, item in enumerate(
        results,
        start=1
    ):

        print(
            f"{i}. {item['title']}"
        )

        print(
            f"   Author : {item['author']}"
        )

        print(
            f"   Score : {item['score']}"
        )

        print()

def search_books_bm25_eval(
    query,
    top_k=10
):

    query = preprocess_text(query)

    tokenized_query = query.split()

    scores = bm25.get_scores(
        tokenized_query
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for idx in top_indices:

        results.append(
            books.iloc[idx]["title"]
        )

    return results