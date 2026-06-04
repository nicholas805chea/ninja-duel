"""Player and shared fighter logic for Ninja Duel: Adaptive Warrior."""


class Fighter:
    """Base class for any combatant in the duel."""

    MAX_HP = 100
    BASE_ATTACK_DAMAGE = 15
    HEAL_AMOUNT = 10
    GRAVITY = 1800
    DEFAULT_MOVE_SPEED = 300
    DEFAULT_JUMP_STRENGTH = 650
    DEFEND_DAMAGE_MULTIPLIER = 0.3

    def __init__(self, name, start_x=0, ground_y=700):
        self.name = name
        self.max_hp = self.MAX_HP
        self.hp = self.max_hp
        self.attack_damage = self.BASE_ATTACK_DAMAGE
        self.heal_amount = self.HEAL_AMOUNT
        self.is_defending = False
        self.defend_timer = 0.0
        self.animation_state = "idle"
        self.alive = True
        self.x = float(start_x)
        self.y = float(ground_y)
        self.ground_y = ground_y
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.on_ground = True
        self.facing = 1
        self.move_speed = self.DEFAULT_MOVE_SPEED
        self.jump_strength = self.DEFAULT_JUMP_STRENGTH

    def reset_turn_state(self):
        """Defend only protects against attacks during one resolved turn."""
        self.is_defending = False
        self.defend_timer = 0.0

    def defend(self, duration=0.65):
        self.is_defending = True
        self.defend_timer = duration

    def update_status_timers(self, dt_seconds):
        if self.defend_timer > 0:
            self.defend_timer = max(0.0, self.defend_timer - dt_seconds)
            if self.defend_timer <= 0:
                self.is_defending = False

    def heal(self):
        """Restore HP without going above the maximum HP."""
        if not self.alive:
            return 0

        old_hp = self.hp
        self.hp = min(self.max_hp, self.hp + self.heal_amount)
        return self.hp - old_hp

    def take_damage(self, damage):
        """Apply damage and return the actual HP lost."""
        if not self.alive:
            return 0

        incoming = damage
        if self.is_defending:
            incoming = max(1, int(round(incoming * self.DEFEND_DAMAGE_MULTIPLIER)))

        old_hp = self.hp
        self.hp = max(0, self.hp - incoming)
        actual_damage = old_hp - self.hp

        if self.hp <= 0:
            self.alive = False
            self.animation_state = "death"
        elif actual_damage > 0:
            self.animation_state = "hit"

        return actual_damage

    def is_dead(self):
        return self.hp <= 0

    def move_left(self, speed_multiplier=1.0):
        if self.alive:
            self.velocity_x = -self.move_speed * speed_multiplier
            self.facing = -1

    def move_right(self, speed_multiplier=1.0):
        if self.alive:
            self.velocity_x = self.move_speed * speed_multiplier
            self.facing = 1

    def stop_horizontal_movement(self):
        self.velocity_x = 0.0

    def jump(self):
        if self.alive and self.on_ground:
            self.velocity_y = -self.jump_strength
            self.on_ground = False

    def apply_physics(self, dt_seconds, min_x, max_x):
        """Move the fighter, apply gravity, and keep them inside the stage."""
        self.x += self.velocity_x * dt_seconds

        if not self.on_ground:
            self.velocity_y += self.GRAVITY * dt_seconds
            self.y += self.velocity_y * dt_seconds

            if self.y >= self.ground_y:
                self.y = float(self.ground_y)
                self.velocity_y = 0.0
                self.on_ground = True

        self.x = max(min_x, min(max_x, self.x))

    def position(self):
        return (int(self.x), int(self.y))


class Player(Fighter):
    """Human-controlled fighter."""

    def __init__(self, start_x=260, ground_y=700):
        super().__init__("Player", start_x=start_x, ground_y=ground_y)
        self.score = 0

    def add_score(self, points):
        self.score += points
