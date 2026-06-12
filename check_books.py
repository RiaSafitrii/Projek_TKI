import joblib

books = joblib.load(
    "models/books.pkl"
)

for title in books["title"].sample(50):
    print(title)