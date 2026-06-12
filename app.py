from flask import Flask
from flask import render_template
from flask import request

from bm25_search import (
    search_books_bm25
)

from search import (
    search_books,
    get_book_by_id
)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search")
def search():

    query = request.args.get("query")

    method = request.args.get("method")

    results = []

    if query:

        if method == "bm25":

            results = search_books_bm25(
                query
            )

        else:

            results = search_books(
                query
            )

    return render_template(
        "results.html",
        query=query,
        method=method,
        results=results
    )

@app.route("/detail/<int:book_id>")
def detail(book_id):

    book = get_book_by_id(book_id)

    return render_template(
        "detail.html",
        book=book
    )

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/dataset")
def dataset():
    return render_template("dataset.html")


@app.route("/evaluation")
def evaluation():
    return render_template("evaluation.html")

if __name__ == "__main__":
    app.run(debug=True)