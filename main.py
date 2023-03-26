from src.cosine_similarity import CosineSimilarity, FuncType

def main():
    cosine_similarity = CosineSimilarity()
    cosine_similarity.run(FuncType.ONE_TO_MANY)

if __name__ == "__main__":
    main()
