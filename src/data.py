import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DATA_DIR, DROP_COLS


def load_dataset_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset_path = DATA_DIR / "high_diamond_ranked_10min.csv"
    raw_dataframe = pd.read_csv(dataset_path)

    features = raw_dataframe.copy()
    features['blueKDA'] = (features['blueKills'] + features['blueAssists']) / (features['blueDeaths'] + 1)
    features['blueGoldPerKill'] = features['blueTotalGold'] / (features['blueKills'] + 1)
    features['blueXPPerMin'] = features['blueTotalExperience'] / 10
    features['blueTotalObjectives'] = features['blueDragons'] + features['blueHeralds'] + features['blueTowersDestroyed']
    features['blueSnowballFactor'] = features['blueGoldDiff'] / (features['blueKills'] + 1)

    drop_columns = ["blueWins"] + [col for col in DROP_COLS if col in features.columns]
    features = features.drop(columns=drop_columns)
    
    targets = raw_dataframe["blueWins"]

    features_train, features_test, targets_train, targets_test = train_test_split(
        features, targets, test_size=0.2, random_state=42, stratify=targets
    )

    feature_scaler = StandardScaler()
    scaled_features_train = pd.DataFrame(
        feature_scaler.fit_transform(features_train),
        columns=features_train.columns,
        index=features_train.index
    )
    scaled_features_test = pd.DataFrame(
        feature_scaler.transform(features_test),
        columns=features_test.columns,
        index=features_test.index
    )

    return scaled_features_train, scaled_features_test, targets_train, targets_test


def load_raw_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Returns unscaled features for display purposes (e.g. real gold/kill values)."""
    dataset_path = DATA_DIR / "high_diamond_ranked_10min.csv"
    raw_dataframe = pd.read_csv(dataset_path)

    features = raw_dataframe.copy()
    features['blueKDA'] = (features['blueKills'] + features['blueAssists']) / (features['blueDeaths'] + 1)
    features['blueGoldPerKill'] = features['blueTotalGold'] / (features['blueKills'] + 1)
    features['blueXPPerMin'] = features['blueTotalExperience'] / 10
    features['blueTotalObjectives'] = features['blueDragons'] + features['blueHeralds'] + features['blueTowersDestroyed']
    features['blueSnowballFactor'] = features['blueGoldDiff'] / (features['blueKills'] + 1)

    drop_columns = ["blueWins"] + [col for col in DROP_COLS if col in features.columns]
    features = features.drop(columns=drop_columns)
    targets = raw_dataframe["blueWins"]

    _, features_test, _, targets_test = train_test_split(
        features, targets, test_size=0.2, random_state=42, stratify=targets
    )
    return features_test, targets_test
