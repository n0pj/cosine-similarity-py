from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import csv
import numpy as np
from numpy import dot
from numpy.linalg import norm

def main():
    data = []

    with open('vehicles-1300.csv', 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    # 特徴量を取得
    # X = np.array([[int(d[8]), float(d[9]), int(d[10]), int(d[11]), int(d[18]), int(d[19]), int(d[20]), int(d[100])] for d in data], dtype=np.float64)

   # メーカー、モデル、車種のカテゴリを取得
    make_categories = set([d[4] for d in data])
    model_categories = set([d[5] for d in data])
    vehicle_categories = set([d[6] for d in data])

    # メーカーの OneHot 表現を取得
    make_encoder = LabelEncoder()
    make_encoder.fit(list(make_categories))
    make_encoded = make_encoder.transform([d[4] for d in data])
    make_ohe = OneHotEncoder()
    make_onehot = make_ohe.fit_transform(make_encoded.reshape(-1, 1))

    # モデルの OneHot 表現を取得
    model_encoder = LabelEncoder()
    model_encoder.fit(list(model_categories))
    model_encoded = model_encoder.transform([d[5] for d in data])
    model_ohe = OneHotEncoder()
    model_onehot = model_ohe.fit_transform(model_encoded.reshape(-1, 1))

    # 車種の OneHot 表現を取得
    vehicle_encoder = LabelEncoder()
    vehicle_encoder.fit(list(vehicle_categories))
    vehicle_encoded = vehicle_encoder.transform([d[6] for d in data])
    vehicle_ohe = OneHotEncoder()
    vehicle_onehot = vehicle_ohe.fit_transform(vehicle_encoded.reshape(-1, 1))

    # エンジンサイズを正規化するための StandardScaler のインスタンスを作成
    engine_size_scaler = StandardScaler()

    # エンジンサイズを正規化
    engine_size_normalized = engine_size_scaler.fit_transform([[d[9]] for d in data])

    # 特徴量のリストに OneHot 表現を追加
    X = np.hstack([make_onehot.toarray(), model_onehot.toarray(), vehicle_onehot.toarray(), engine_size_normalized])

    # for i in range(len(X)):
    #     for j in range(i+1, len(X)):
    #         similarity = dot(X[i], X[j].T) / (norm(X[i]) * norm(X[j]))
    #         print(f"車両{i+1}と車両{j+1}の類似度：{similarity}")
    #     break

    similarity_dict = {}

    for i in range(len(X)):
        for j in range(i+1, len(X)):
            similarity = dot(X[i], X[j].T) / (norm(X[i]) * norm(X[j]))
            similarity_dict[(i+1, j+1)] = similarity

    ranked_similarities = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)

    # reversed_ranked_similarities = ranked_similarities[::-1]

    for i, (pair, similarity) in enumerate(ranked_similarities):
        print(f"{i+1}. 車両{pair[0]}と車両{pair[1]}の類似度：{similarity}")

if __name__ == "__main__":
    main()
