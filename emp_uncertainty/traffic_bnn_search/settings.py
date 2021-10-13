SETTINGS = [
    {"cf": 6, "cd": 2, "opt": "sgd", "lr": 0.01, "mom": 0.0},
    # Variation in learning rate
    {"cf": 6, "cd": 2, "opt": "sgd", "lr": 0.05, "mom": 0.0},
    {"cf": 6, "cd": 2, "opt": "sgd", "lr": 0.1, "mom": 0.0},
    # Variation in momentum
    {"cf": 6, "cd": 2, "opt": "sgd", "lr": 0.01, "mom": 0.1},
    # Variation in optimizer
    {"cf": 6, "cd": 2, "opt": "RMSProp", "lr": 0.001, "mom": 0.0},
    # Variation in number of flipout layers
    {"cf": 4, "cd": 2, "opt": "sgd", "lr": 0.01, "mom": 0.0},
    {"cf": 2, "cd": 2, "opt": "sgd", "lr": 0.01, "mom": 0.0},
    {"cf": 6, "cd": 1, "opt": "sgd", "lr": 0.01, "mom": 0.0},
    {"cf": 4, "cd": 1, "opt": "sgd", "lr": 0.01, "mom": 0.0},
    {"cf": 2, "cd": 1, "opt": "sgd", "lr": 0.01, "mom": 0.0},
]