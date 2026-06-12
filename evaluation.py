from search import search_books_eval

from bm25_search import (
    search_books_bm25_eval
)

from ground_truth import (
    ground_truth
)


def precision_at_k(
    retrieved,
    relevant,
    k=10
):

    retrieved = retrieved[:k]

    relevant_found = sum(
        1
        for doc in retrieved
        if doc in relevant
    )

    return relevant_found / k


print("\n=== EVALUASI TF-IDF ===\n")

total_precision = 0

for query, relevant_docs in ground_truth.items():

    retrieved_docs = search_books_eval(
        query
    )

    precision = precision_at_k(
        retrieved_docs,
        relevant_docs
    )

    total_precision += precision

    print(
        f"{query} -> Precision@10 = {precision:.2f}"
    )

mean_precision = (
    total_precision /
    len(ground_truth)
)

print(
    f"\nMean Precision TF-IDF = {mean_precision:.2f}"
)

print("\n========================\n")

print("\n=== EVALUASI BM25 ===\n")

total_precision = 0

for query, relevant_docs in ground_truth.items():

    retrieved_docs = (
        search_books_bm25_eval(
            query
        )
    )

    precision = precision_at_k(
        retrieved_docs,
        relevant_docs
    )

    total_precision += precision

    print(
        f"{query} -> Precision@10 = {precision:.2f}"
    )

mean_precision = (
    total_precision /
    len(ground_truth)
)

print(
    f"\nMean Precision BM25 = {mean_precision:.2f}"
)