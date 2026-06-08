"""Adaptive enemy AI for Ninja Duel: Adaptive Warrior.

Uses a Multi-Layer Perceptron (MLP) neural network to predict the player's
next action from a 12-dimensional game-state feature vector, then selects
an optimal counter-action.
"""

import random

import numpy as np


class MLPPredictor:
    """Multi-Layer Perceptron for predicting player actions.

    Architecture: Input(12) -> Hidden1(16, ReLU) -> Hidden2(8, ReLU) -> Output(3, Softmax)
    Trained online via backpropagation with categorical cross-entropy loss.
    """

    def __init__(self, learning_rate=0.03):
        self.learning_rate = learning_rate
        self.input_dim = 12
        self.hidden1_dim = 16
        self.hidden2_dim = 8
        self.output_dim = 3

        # He Initialization for ReLU layers
        self.W1 = np.random.randn(self.hidden1_dim, self.input_dim) * np.sqrt(
            2.0 / self.input_dim
        )
        self.b1 = np.zeros((self.hidden1_dim, 1))

        self.W2 = np.random.randn(self.hidden2_dim, self.hidden1_dim) * np.sqrt(
            2.0 / self.hidden1_dim
        )
        self.b2 = np.zeros((self.hidden2_dim, 1))

        self.W3 = np.random.randn(self.output_dim, self.hidden2_dim) * np.sqrt(
            2.0 / self.hidden2_dim
        )
        self.b3 = np.zeros((self.output_dim, 1))

        # Training memory buffer
        self.experience_buffer = []

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        shift_Z = Z - np.max(Z, axis=0, keepdims=True)  # Numerical stability
        exps = np.exp(shift_Z)
        return exps / np.sum(exps, axis=0, keepdims=True)

    def forward(self, X):
        """Forward pass. X is a 12-dimensional feature list or array."""
        a0 = np.array(X).reshape(-1, 1)

        z1 = np.dot(self.W1, a0) + self.b1
        a1 = self.relu(z1)

        z2 = np.dot(self.W2, a1) + self.b2
        a2 = self.relu(z2)

        z3 = np.dot(self.W3, a2) + self.b3
        a3 = self.softmax(z3)

        return a3.flatten(), (a0, z1, a1, z2, a2, z3, a3)

    def record_experience(self, features, actual_action):
        """Log state features and the player's chosen action for training."""
        action_map = {"Attack": 0, "Defend": 1, "Heal": 2}
        if actual_action in action_map:
            target_idx = action_map[actual_action]
            y_one_hot = np.zeros((3, 1))
            y_one_hot[target_idx] = 1.0
            self.experience_buffer.append((features, y_one_hot))

    def train_on_buffer(self):
        """Train the network using backpropagation on all recorded experiences."""
        if len(self.experience_buffer) < 10:
            return  # Need sufficient data points

        for features, y_true in self.experience_buffer:
            # Forward pass
            _, cache = self.forward(features)
            a0, z1, a1, z2, a2, z3, a3 = cache

            # Backpropagation
            # Output layer error (cross-entropy + softmax gradient)
            delta3 = a3 - y_true
            dW3 = np.dot(delta3, a2.T)
            db3 = delta3

            # Hidden layer 2 error
            delta2 = np.dot(self.W3.T, delta3) * self.relu_derivative(z2)
            dW2 = np.dot(delta2, a1.T)
            db2 = delta2

            # Hidden layer 1 error
            delta1 = np.dot(self.W2.T, delta2) * self.relu_derivative(z1)
            dW1 = np.dot(delta1, a0.T)
            db1 = delta1

            # Gradient descent parameter updates
            self.W3 -= self.learning_rate * dW3
            self.b3 -= self.learning_rate * db3
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1

        # Clear buffer after learning pass
        self.experience_buffer.clear()


class AdaptiveAI:
    """Predicts the player's next action using an MLP neural network and
    selects an optimal counter-action.

    The MLP takes a 12-dimensional feature vector (HP ratios, distance,
    airborne/defending states, boss phase flag, and the last two player
    actions) and outputs a probability distribution over {Attack, Defend,
    Heal}. The AI then applies counter-tactics based on the prediction.
    """

    ACTIONS = ("Attack", "Defend", "Heal")

    def __init__(self):
        self.mlp = MLPPredictor(learning_rate=0.03)
        self.action_history = []
        self.turn_count = 0
        self.last_prediction = "None"
        self._jump_cooldown = 0.0
        self._cached_features = None

    def _encode_features(self, player, enemy, distance, boss_phase):
        """Build the 12-dimensional input feature vector for the MLP.

        Features:
            x1:  Player HP ratio          (0.0 – 1.0)
            x2:  Enemy HP ratio           (0.0 – 1.0)
            x3:  Normalised distance      (0.0 – 1.0)
            x4:  Player airborne flag     (0.0 or 1.0)
            x5:  Player defending flag    (0.0 or 1.0)
            x6:  Boss phase flag          (0.0 or 1.0)
            x7–x9:   Last action one-hot  (Attack, Defend, Heal)
            x10–x12: Second-last action one-hot
        """
        player_hp = player.hp / 100.0
        enemy_hp = enemy.hp / 100.0
        norm_dist = min(distance / 1000.0, 1.0)
        p_airborne = 1.0 if not player.on_ground else 0.0
        p_defending = 1.0 if player.is_defending else 0.0
        boss_active = 1.0 if boss_phase else 0.0

        action_map = {"Attack": [1, 0, 0], "Defend": [0, 1, 0], "Heal": [0, 0, 1]}
        h1 = (
            action_map.get(self.action_history[-1], [0, 0, 0])
            if len(self.action_history) >= 1
            else [0, 0, 0]
        )
        h2 = (
            action_map.get(self.action_history[-2], [0, 0, 0])
            if len(self.action_history) >= 2
            else [0, 0, 0]
        )

        return [
            player_hp, enemy_hp, norm_dist,
            p_airborne, p_defending, boss_active,
        ] + h1 + h2

    def record_player_action(self, action, player=None, enemy=None, boss_phase=False):
        """Record the player's action and train the MLP when enough data exists."""
        self.action_history.append(action)

        # Compute features from the current game state for training
        if player is not None and enemy is not None:
            distance = abs(player.x - enemy.x)
            features = self._encode_features(player, enemy, distance, boss_phase)
            self.mlp.record_experience(features, action)
        elif self._cached_features is not None:
            self.mlp.record_experience(self._cached_features, action)

        # Train when enough experiences have been collected
        if len(self.mlp.experience_buffer) >= 10:
            self.mlp.train_on_buffer()

    def predict_player_action(self):
        """Use the MLP forward pass to predict the player's next action.

        During the first three turns (learning phase), returns 'Learning'
        since the network has insufficient training data.
        """
        if self.turn_count <= 3 or self._cached_features is None:
            self.last_prediction = "Learning"
            return self.last_prediction

        probs, _ = self.mlp.forward(self._cached_features)
        actions = ["Attack", "Defend", "Heal"]
        predicted_idx = int(np.argmax(probs))
        self.last_prediction = actions[predicted_idx]
        return self.last_prediction

    def choose_action(self, boss_phase=False):
        """Choose the enemy action (turn-based fallback).

        First three turns are random. After that, the MLP predicts
        the player's most likely action and a counter is selected.
        """
        self.turn_count += 1

        if self.turn_count <= 3:
            self.last_prediction = "Learning"
            if boss_phase:
                return random.choices(self.ACTIONS, weights=(4, 1, 1), k=1)[0]
            return random.choice(self.ACTIONS)

        prediction = self.predict_player_action()

        if prediction == "Attack":
            return "Attack" if boss_phase and random.random() < 0.35 else "Defend"

        if prediction == "Heal":
            return "Attack"

        if prediction == "Defend":
            return random.choices(
                ("Attack", "Heal"),
                weights=(4, 1) if boss_phase else (1, 1),
                k=1,
            )[0]

        return random.choice(self.ACTIONS)

    def choose_autonomous_action(self, enemy, player, distance, boss_phase=False):
        """Choose an enemy action for real-time combat using MLP predictions.

        Combines the neural network's predicted player action with distance,
        HP levels, boss phase, and player defending state to select the best
        counter-action.
        """
        attack_range = 130

        # Update cached features every frame for prediction and training
        self._cached_features = self._encode_features(
            player, enemy, distance, boss_phase
        )

        if distance > attack_range:
            if enemy.hp <= 35 and enemy.hp < enemy.max_hp and random.random() < 0.35:
                self.last_prediction = self.predict_player_action()
                return "Heal"
            return None

        self.turn_count += 1
        if self.turn_count <= 3:
            self.last_prediction = "Learning"
            return random.choices(
                self.ACTIONS,
                weights=(5, 2, 1) if boss_phase else (3, 2, 1),
                k=1,
            )[0]

        prediction = self.predict_player_action()

        if enemy.hp <= 35 and enemy.hp < enemy.max_hp:
            if distance > attack_range * 0.65 or player.is_defending:
                if random.random() < (0.45 if boss_phase else 0.65):
                    return "Heal"

        if prediction == "Attack" and distance < attack_range:
            return random.choices(
                ("Attack", "Defend"),
                weights=(4, 2) if boss_phase else (2, 5),
                k=1,
            )[0]

        if player.is_defending:
            return random.choices(
                ("Attack", "Heal"),
                weights=(4, 1) if boss_phase else (2, 1),
                k=1,
            )[0]

        if prediction == "Heal":
            return "Attack"

        if prediction == "Defend":
            return random.choices(
                ("Attack", "Heal"),
                weights=(5, 1) if boss_phase else (3, 1),
                k=1,
            )[0]

        return random.choices(
            ("Attack", "Defend", "Heal"),
            weights=(6, 1, 1) if boss_phase else (4, 2, 1),
            k=1,
        )[0]

    def update_movement_timer(self, dt_seconds):
        if self._jump_cooldown > 0:
            self._jump_cooldown = max(0.0, self._jump_cooldown - dt_seconds)

    def choose_movement(
        self, enemy, player, predicted_action, boss_phase=False, enemy_action=None
    ):
        """Choose movement from distance and the current MLP prediction.

        Checks spacing first, then uses the predicted player action to decide
        whether to close distance, retreat, hold position, or occasionally jump.
        """
        distance = abs(enemy.x - player.x)
        attack_range = 130
        retreat_distance = 220 if not boss_phase else 140
        close_distance = attack_range - (15 if boss_phase else 0)

        near_left_edge = enemy.x < 140
        near_right_edge = enemy.x > 860
        stuck_near_player = distance < 90

        if (
            enemy.on_ground
            and self._jump_cooldown <= 0
            and (near_left_edge or near_right_edge or stuck_near_player)
        ):
            self._jump_cooldown = 1.4
            return "jump"

        if distance > close_distance:
            return "approach"

        if (
            predicted_action == "Attack"
            and distance < retreat_distance
            and not boss_phase
        ):
            return "retreat"

        if enemy_action == "Heal" and distance < retreat_distance:
            return "retreat"

        if enemy.hp <= 35 and distance < retreat_distance:
            return "retreat"

        return "hold"
