"""Adaptive enemy AI for Ninja Duel: Adaptive Warrior."""

import random


class AdaptiveAI:
    """Predicts the player action from frequency counts and counters it.

    This is intentionally simple and explainable: no machine learning library is
    used. The AI stores every player action, counts how often each action was
    chosen, predicts the most common action, then chooses a counter action.
    """

    ACTIONS = ("Attack", "Defend", "Heal")

    def __init__(self):
        self.attack_count = 0
        self.defend_count = 0
        self.heal_count = 0
        self.history = []
        self.turn_count = 0
        self.last_prediction = "None"
        self._jump_cooldown = 0.0

    def record_player_action(self, action):
        """Update the AI memory after each player turn."""
        self.history.append(action)

        if action == "Attack":
            self.attack_count += 1
        elif action == "Defend":
            self.defend_count += 1
        elif action == "Heal":
            self.heal_count += 1

    def predict_player_action(self):
        """Return the most common player action seen so far.

        If two or more actions are tied, a tied action is selected randomly so
        the enemy does not become completely predictable during equal counts.
        """
        counts = {
            "Attack": self.attack_count,
            "Defend": self.defend_count,
            "Heal": self.heal_count,
        }
        highest_count = max(counts.values())

        if highest_count == 0:
            self.last_prediction = "None"
            return self.last_prediction

        tied_actions = [
            action for action, count in counts.items() if count == highest_count
        ]
        self.last_prediction = random.choice(tied_actions)
        return self.last_prediction

    def choose_action(self, boss_phase=False):
        """Choose the enemy action.

        First three turns are random, as required. After that, the enemy predicts
        the player's most common behavior and responds with a counter strategy:
        predicted Attack -> Defend, predicted Heal -> Attack, predicted Defend ->
        Attack or Heal. Boss phase keeps the same idea but favors Attack more.
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
        """Choose an enemy action for real-time combat.

        The enemy does not wait for the player to press an action key. It still
        uses the same adaptive prediction counts, but now combines them with
        distance, HP, boss phase, and whether the player is defending.
        """
        attack_range = 130

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
        """Choose movement from distance and the current adaptive prediction.

        The AI remains explainable: it does not use machine learning. It checks
        spacing first, then uses the predicted player action to decide whether
        to close distance, retreat, hold position, or occasionally jump.
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
