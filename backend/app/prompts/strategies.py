NON_IMPOSTER_CLUE_STRATEGIES: tuple[dict[str, str], ...] = (
    {
        "name": "Indirect association",
        "prompt": (
            "Write a phrase that is indirectly related to the secret word, not a "
            "definition or obvious synonym. Use an association that a teammate who "
            "knows the word can connect, while an imposter would find ambiguous. "
            "Examples: satellite -> night dish; library -> quiet checkout; "
            "volcano -> buried pressure."
        ),
    },
    {
        "name": "Side effect",
        "prompt": (
            "Write a phrase based on something caused by, enabled by, or resulting "
            "from the secret word. Do not describe the object directly. Prefer a "
            "consequence or outcome someone could infer from the word. "
            "Examples: camera -> frozen memory; bridge -> shorter commute; "
            "piano -> room applause."
        ),
    },
    {
        "name": "Context scene",
        "prompt": (
            "Write a phrase that evokes a place, scene, or situation where the "
            "secret word naturally appears. The clue should feel concrete but not "
            "name the word or define it. "
            "Examples: guitar -> campfire chorus; island -> ferry arrival; "
            "theater -> velvet seats."
        ),
    },
    {
        "name": "Hidden detail",
        "prompt": (
            "Write a phrase using a specific detail someone who knows the secret "
            "word may recognize. Avoid broad categories and obvious descriptions. "
            "Pick a small visual, sensory, or functional detail. "
            "Examples: rocket -> countdown plume; forest -> mossy fallen log; "
            "window -> morning condensation."
        ),
    },
    {
        "name": "Contrast hint",
        "prompt": (
            "Write a phrase based on something the secret word is often compared "
            "with or contrasted against. The clue should point through contrast, "
            "not by direct definition. "
            "Examples: desert -> ocean opposite; mountain -> inverted valley; "
            "apple -> orange rival."
        ),
    },
    {
        "name": "Specification",
        "prompt": (
            "Write a phrase naming a specific instance, person, brand, place, or "
            "example strongly associated with the secret word. The clue should be "
            "specific enough for teammates to connect, but avoid naming the secret "
            "word itself. Examples: social media -> Mark Zuckerberg; online "
            "marketplace -> Jeff Bezos; electric car -> Elon Musk."
        ),
    },
)
