import joblib
from pathlib import Path
from functools import lru_cache

class VotingModelPredictor:
    def __init__(self, model_path: Path):
        self.model = joblib.load(model_path)

    def predict_probabilities(
        self,
        feature_matrix: list[list[float]],
    ) -> list[float]:
        probabilities = self.model.predict_proba(feature_matrix)
        classes = list(self.model.classes_)
        positive_index = classes.index(1)

        return [
            float(row[positive_index])
            for row in probabilities
        ]
    
    def score_candidates(
        self,
        player_ids: list[str],
        feature_matrix: list[list[float]],
    ) -> dict[str, float]:
        probabilities = self.predict_probabilities(feature_matrix)

        return dict(zip(player_ids, probabilities, strict=True))
    

@lru_cache
def get_voting_predictor(model_path: str) -> VotingModelPredictor:
    return VotingModelPredictor(Path(model_path))