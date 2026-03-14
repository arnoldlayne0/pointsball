from pointsball.models.predictor import TARGET_COL, build_training_data, train


def test_build_training_data_shape(fpl_dataset_fixture):
    df, feature_cols = build_training_data(fpl_dataset_fixture)
    # Must have rows remaining after dropping all-null rolling rows
    assert len(df) > 0
    # Must include the target and at least some rolling features
    assert TARGET_COL in df.columns
    assert any("rolling" in c for c in feature_cols)
    # Target must NOT be in feature_cols
    assert TARGET_COL not in feature_cols


def test_train_returns_model_and_feature_cols(fpl_dataset_fixture):
    model, feature_cols = train(fpl_dataset_fixture)
    assert len(feature_cols) > 0
    # Model can make predictions on a numpy array with the right number of columns
    df, _ = build_training_data(fpl_dataset_fixture)
    X = df.select(feature_cols).to_numpy()
    preds = model.predict(X)
    assert len(preds) == len(df)
