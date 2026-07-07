from app.services.agents import strategy_loader


def test_imposter_strategy_weights_follow_previous_clue_count_table() -> None:
    assert strategy_loader._imposter_strategy_weights_for_previous_clues(0) == {
        "Abstraction": 50,
        "Adjacent association": 50,
    }
    assert strategy_loader._imposter_strategy_weights_for_previous_clues(1) == {
        "Abstraction": 40,
        "Adjacent association": 40,
        "Ride previous clues": 10,
        "Contextual guess": 10,
    }
    assert strategy_loader._imposter_strategy_weights_for_previous_clues(2) == {
        "Abstraction": 25,
        "Adjacent association": 25,
        "Ride previous clues": 20,
        "Contextual guess": 20,
        "Cluster matching": 10,
    }
    assert strategy_loader._imposter_strategy_weights_for_previous_clues(3) == {
        "Abstraction": 10,
        "Adjacent association": 10,
        "Ride previous clues": 30,
        "Contextual guess": 30,
        "Cluster matching": 20,
    }
    assert strategy_loader._imposter_strategy_weights_for_previous_clues(4) == {
        "Abstraction": 5,
        "Adjacent association": 5,
        "Ride previous clues": 25,
        "Contextual guess": 25,
        "Cluster matching": 40,
    }


def test_imposter_strategy_weights_clamp_out_of_range_counts() -> None:
    assert strategy_loader._imposter_strategy_weights_for_previous_clues(-1) == (
        strategy_loader._imposter_strategy_weights_for_previous_clues(0)
    )
    assert strategy_loader._imposter_strategy_weights_for_previous_clues(99) == (
        strategy_loader._imposter_strategy_weights_for_previous_clues(4)
    )


def test_assign_imposter_strategy_samples_from_weighted_candidates(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_choices(population, weights, k):
        captured["population"] = population
        captured["weights"] = weights
        captured["k"] = k
        return [population[0]]

    monkeypatch.setattr(strategy_loader.random, "choices", fake_choices)

    selected_strategy = strategy_loader.assign_imposter_clue_strategy(
        "few_shot",
        previous_clue_count=4,
    )

    assert selected_strategy["name"] == "Ride previous clues"
    assert captured["k"] == 1
    assert [strategy["name"] for strategy in captured["population"]] == [
        "Ride previous clues",
        "Abstraction",
        "Cluster matching",
        "Contextual guess",
        "Adjacent association",
    ]
    assert captured["weights"] == [25, 5, 40, 25, 5]


def test_assign_non_imposter_strategy_keeps_uniform_random_choice(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_choice(strategies):
        captured["strategies"] = strategies
        return strategies[0]

    monkeypatch.setattr(strategy_loader.random, "choice", fake_choice)

    selected_strategy = strategy_loader.assign_non_imposter_clue_strategy("few_shot")

    assert selected_strategy["name"] == "Indirect association"
    assert len(captured["strategies"]) == 6
