"""Enemy fighter with boss phase behavior."""

from player import Fighter


class Enemy(Fighter):
    """AI-controlled enemy warrior."""

    BOSS_HP_THRESHOLD = 30
    BOSS_ATTACK_DAMAGE = 25

    def __init__(self, start_x=740, ground_y=700):
        super().__init__("Enemy", start_x=start_x, ground_y=ground_y)
        self.boss_phase = False

    def update_boss_phase(self):
        """Activate boss mode once when HP falls below the threshold."""
        if not self.boss_phase and 0 < self.hp < self.BOSS_HP_THRESHOLD:
            self.boss_phase = True
            self.attack_damage = self.BOSS_ATTACK_DAMAGE
            return True
        return False
