import random

def generate_feature_vector():
    # Simulate 78 features between 0.5 to 1.0 (like confidence scores or normalized values)
    return [round(random.uniform(0.5, 1.0), 2) for _ in range(78)]
