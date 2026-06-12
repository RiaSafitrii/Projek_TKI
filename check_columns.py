import joblib

books = joblib.load("models/books.pkl")

print(books.columns.tolist())