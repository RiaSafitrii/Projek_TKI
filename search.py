import joblib

from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import preprocess_text

print("Memuat model...")

vectorizer = joblib.load(
    "models/vectorizer.pkl"
)

tfidf_matrix = joblib.load(
    "models/tfidf_matrix.pkl"
)

books = joblib.load(
    "models/books.pkl"
)

print("Model berhasil dimuat.")


def search_books(query, top_k=10):

    query = preprocess_text(query)

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    sorted_indices = similarities.argsort()[::-1]

    results = []

    for idx in sorted_indices:

        # Hanya tampilkan dokumen yang relevan
        if similarities[idx] > 0:

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
                str(
                    books.iloc[idx]["description"]
                )[:250],

                "score":
                round(
                    float(similarities[idx]) * 100,
                    2
                )

            })

        if len(results) >= top_k:
            break

    return results


def search_books_eval(query, top_k=10):

    query = preprocess_text(query)

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    sorted_indices = similarities.argsort()[::-1]

    results = []

    for idx in sorted_indices:

        if similarities[idx] > 0:

            results.append(
                books.iloc[idx]["title"]
            )

        if len(results) >= top_k:
            break

    return results


def get_book_by_id(book_id):

    result = books[
        books["bookId"].astype(str) == str(book_id)
    ]

    if len(result) == 0:
        return None

    book = result.iloc[0]

    return {
        "bookId": book["bookId"],
        "title": book["title"],
        "author": book["author"],
        "genres": book["genres"],
        "description": book["description"],
        "rating": book["avg_rating"]
    }


if __name__ == "__main__":

    query = input("\nMasukkan Query: ")

    results = search_books(query)

    print("\nHASIL PENCARIAN\n")

    if len(results) == 0:

        print("Tidak ditemukan buku yang relevan.")

    else:

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
                f"   Rating : {item['rating']}"
            )

            print(
                f"   Score  : {item['score']}"
            )

            print()