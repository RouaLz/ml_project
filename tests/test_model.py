from model_pipeline import prepare_data


def test_prepare_data():
    X_train, X_test, y_train, y_test, scaler = prepare_data("Churn_Modelling.csv")
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert X_train.shape[1] == X_test.shape[1]
