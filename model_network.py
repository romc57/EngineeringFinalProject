from sklearn.neighbors import KNeighborsClassifier


def trainKnn(matrix, tags, neighbors):
    model = KNeighborsClassifier(neighbors)
    return model.fit(matrix, tags)


def predict(model, data):
    return model.predict(data)


