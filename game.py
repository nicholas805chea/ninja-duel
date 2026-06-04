"""Main game controller for Ninja Duel: Adaptive Warrior."""

from pathlib import Path

import pygame

from ai import AdaptiveAI
from animation import FighterAnimator
from enemy import Enemy
from player import Player
from ui import UI


class Game:
    WIDTH = 1000
    HEIGHT = 720
    FPS = 60
    HIGH_SCORE_FILE = Path("high_score.txt")
    GROUND_Y = 700
    STAGE_MIN_X = 80
    STAGE_MAX_X = 920
    MIN_FIGHTER_SPACING = 85
    ATTACK_RANGE = 130
    PLAYER_ACTION_COOLDOWN = 0.55
    ENEMY_ACTION_COOLDOWN = 0.75
    BOSS_ENEMY_ACTION_COOLDOWN = 0.5
    DEFEND_DURATION = 0.65

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Ninja Duel: Adaptive Warrior")
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.ui = UI(self.screen)

        self.player = Player(start_x=260, ground_y=self.GROUND_Y)
        self.enemy = Enemy(start_x=740, ground_y=self.GROUND_Y)
        self.ai = AdaptiveAI()
        self.high_score = self.load_high_score()
        self.battle_log = ["Move:1/3  Jump:2  Atk:8  Def:9  Heal:0"]
        self.game_over = False
        self.winner = None
        self.enemy_current_action = None
        self.player_action_cooldown = 0.0
        self.enemy_action_cooldown = 0.0
        self.pending_player_action = None
        # Track which movement keys are currently held (via KEYDOWN/KEYUP events).
        self.held_keys = set()

        sprite_dir = Path("Martial Hero") / "Sprites"
        self.player_animator = FighterAnimator(sprite_dir, self.player.position())
        self.enemy_animator = FighterAnimator(sprite_dir, self.enemy.position(), flip=True)

    def load_high_score(self):
        try:
            return int(self.HIGH_SCORE_FILE.read_text(encoding="utf-8").strip())
        except (FileNotFoundError, ValueError):
            return 0

    def save_high_score(self):
        if self.player.score > self.high_score:
            self.high_score = self.player.score
            self.HIGH_SCORE_FILE.write_text(str(self.high_score), encoding="utf-8")

    def run(self):
        running = True
        while running:
            dt_ms = self.clock.tick(self.FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r and self.game_over:
                        self.restart()
                    # Action keys — single-shot via event.
                    elif event.key in (pygame.K_8, pygame.K_KP8):
                        self.pending_player_action = "Attack"
                    elif event.key in (pygame.K_9, pygame.K_KP9):
                        self.pending_player_action = "Defend"
                    elif event.key in (pygame.K_0, pygame.K_KP0):
                        self.pending_player_action = "Heal"
                    # Movement keys — track held state.
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3,
                                       pygame.K_KP1, pygame.K_KP2, pygame.K_KP3):
                        self.held_keys.add(event.key)
                elif event.type == pygame.KEYUP:
                    self.held_keys.discard(event.key)
                elif event.type == pygame.WINDOWFOCUSLOST:
                    # Clear held keys so no phantom movement when focus returns.
                    self.held_keys.clear()

            self.update(dt_ms)
            self.draw()

        self.save_high_score()
        pygame.quit()

    def restart(self):
        self.player = Player(start_x=260, ground_y=self.GROUND_Y)
        self.enemy = Enemy(start_x=740, ground_y=self.GROUND_Y)
        self.ai = AdaptiveAI()
        self.battle_log = ["Move:1/3  Jump:2  Atk:8  Def:9  Heal:0"]
        self.game_over = False
        self.winner = None
        self.enemy_current_action = None
        self.player_action_cooldown = 0.0
        self.enemy_action_cooldown = 0.0
        self.pending_player_action = None
        self.held_keys.clear()
        self.player_animator.set_state("idle")
        self.enemy_animator.set_state("idle")
        self._sync_animators_to_fighters()

    def _fighters_in_attack_range(self):
        return abs(self.player.x - self.enemy.x) <= self.ATTACK_RANGE

    def perform_player_action(self, action):
        """Execute a player action. Returns True if the action was consumed."""
        if self.game_over or self.player_action_cooldown > 0:
            return False

        self.player.stop_horizontal_movement()
        self.player_action_cooldown = self.PLAYER_ACTION_COOLDOWN
        self.ai.record_player_action(action)
        self._perform_action(
            action,
            self.player,
            self.enemy,
            "Player",
            "Enemy",
            self.player_animator,
            self.enemy_animator,
            score_for_damage=True,
        )
        return True

    def perform_enemy_action(self, action):
        if self.game_over or not action:
            return

        self.enemy_current_action = action
        self.enemy_action_cooldown = (
            self.BOSS_ENEMY_ACTION_COOLDOWN
            if self.enemy.boss_phase
            else self.ENEMY_ACTION_COOLDOWN
        )
        self._perform_action(
            action,
            self.enemy,
            self.player,
            "Enemy",
            "Player",
            self.enemy_animator,
            self.player_animator,
            score_for_damage=False,
        )

    def _perform_action(
        self,
        action,
        actor,
        target,
        actor_label,
        target_label,
        actor_animator,
        target_animator,
        score_for_damage=False,
    ):
        actor.animation_state = "idle"

        if action == "Attack":
            actor_animator.set_state("attack")
            in_range = abs(actor.x - target.x) <= self.ATTACK_RANGE
            target_airborne = not target.on_ground

            if not in_range:
                self.add_log(f"{actor_label} attack missed: out of range.")
            elif target_airborne:
                self.add_log(f"{target_label} dodged by jumping!")
            else:
                damage = target.take_damage(actor.attack_damage)
                if score_for_damage and damage > 0:
                    self.player.add_score(10)
                self.add_log(f"{actor_label} dealt {damage} damage.")
                self._sync_combat_animation(target, target_animator)

        elif action == "Defend":
            actor.defend(self.DEFEND_DURATION)
            self.add_log(f"{actor_label} defended.")

        elif action == "Heal":
            healed = actor.heal()
            if healed:
                self.add_log(f"{actor_label} healed {healed} HP.")
            else:
                self.add_log(f"{actor_label} tried to heal, but HP is full.")

        if actor_label == "Player":
            self.add_log(f"AI prediction: {self.ai.last_prediction}")

        boss_activated = self.enemy.update_boss_phase()
        if boss_activated:
            self.add_log("BOSS PHASE ACTIVATED")

        self._sync_combat_animation(actor, actor_animator)
        self.check_battle_result()

    def _sync_combat_animation(self, fighter, animator):
        if fighter.is_dead():
            animator.set_state("death")
        elif fighter.animation_state == "hit":
            animator.set_state("hit")

    def add_log(self, message):
        self.battle_log.append(message)
        self.battle_log = self.battle_log[-5:]

    def check_battle_result(self):
        if self.player.is_dead() and self.enemy.is_dead():
            self.game_over = True
            self.winner = "Draw"
            self.add_log("The duel ends in a draw.")
        elif self.enemy.is_dead():
            self.game_over = True
            self.winner = "Player"
            self.player.add_score(100)
            self.add_log("You win! +100 score.")
        elif self.player.is_dead():
            self.game_over = True
            self.winner = "Enemy"
            self.add_log("You lost the duel.")

        if self.game_over:
            self.save_high_score()

    def update(self, dt_ms):
        dt_seconds = dt_ms / 1000
        self._update_timers(dt_seconds)

        if not self.game_over:
            self._handle_player_controls()
            self._handle_enemy_combat()
            self._handle_enemy_movement(dt_seconds)
            self._apply_fighter_physics(dt_seconds)
            self._update_facing()
        else:
            self.player.stop_horizontal_movement()
            self.enemy.stop_horizontal_movement()

        self._sync_animators_to_fighters()
        self._update_movement_animation_states()
        self.player_animator.update(dt_ms)
        self.enemy_animator.update(dt_ms)

    def _update_timers(self, dt_seconds):
        self.player_action_cooldown = max(
            0.0, self.player_action_cooldown - dt_seconds
        )
        self.enemy_action_cooldown = max(0.0, self.enemy_action_cooldown - dt_seconds)
        self.player.update_status_timers(dt_seconds)
        self.enemy.update_status_timers(dt_seconds)

    def _handle_player_controls(self):
        # Consume a pending action (set by KEYDOWN event, single-shot).
        # Only block movement if the action was actually executed this frame.
        action = self.pending_player_action
        self.pending_player_action = None

        if action and self.perform_player_action(action):
            # Action executed: cancel movement for this one frame only.
            self.player.stop_horizontal_movement()
            return

        # Movement uses held_keys (tracked via KEYDOWN/KEYUP events).
        # 1 = left, 2 = jump, 3 = right  (also numpad 1/2/3)
        moving_left  = pygame.K_1 in self.held_keys or pygame.K_KP1 in self.held_keys
        moving_right = pygame.K_3 in self.held_keys or pygame.K_KP3 in self.held_keys
        jumping      = pygame.K_2 in self.held_keys or pygame.K_KP2 in self.held_keys

        if moving_left and not moving_right:
            self.player.move_left()
        elif moving_right and not moving_left:
            self.player.move_right()
        else:
            self.player.stop_horizontal_movement()

        if jumping:
            self.player.jump()

    def _pressed_player_action(self, keys):  # kept for reference, no longer used
        if keys[pygame.K_8] or keys[pygame.K_KP8]:
            return "Attack"
        if keys[pygame.K_9] or keys[pygame.K_KP9]:
            return "Defend"
        if keys[pygame.K_0] or keys[pygame.K_KP0]:
            return "Heal"
        return None

    def _handle_enemy_combat(self):
        if self.enemy_action_cooldown > 0:
            return

        distance = abs(self.enemy.x - self.player.x)
        action = self.ai.choose_autonomous_action(
            self.enemy,
            self.player,
            distance,
            boss_phase=self.enemy.boss_phase,
        )
        if action:
            self.perform_enemy_action(action)
        else:
            self.enemy_current_action = None

    def _handle_enemy_movement(self, dt_seconds):
        self.ai.update_movement_timer(dt_seconds)
        prediction = self.ai.last_prediction
        if self.ai.turn_count > 3:
            prediction = self.ai.predict_player_action()

        movement = self.ai.choose_movement(
            self.enemy,
            self.player,
            prediction,
            boss_phase=self.enemy.boss_phase,
            enemy_action=self.enemy_current_action,
        )
        speed_multiplier = 1.2 if self.enemy.boss_phase else 0.85

        if movement == "approach":
            self._move_enemy_toward_player(speed_multiplier)
        elif movement == "retreat":
            self._move_enemy_away_from_player(0.95)
        elif movement == "jump":
            self.enemy.jump()
            self._move_enemy_toward_player(speed_multiplier * 0.7)
        else:
            self.enemy.stop_horizontal_movement()

    def _move_enemy_toward_player(self, speed_multiplier):
        if self.enemy.x > self.player.x:
            self.enemy.move_left(speed_multiplier)
        else:
            self.enemy.move_right(speed_multiplier)

    def _move_enemy_away_from_player(self, speed_multiplier):
        if self.enemy.x > self.player.x:
            self.enemy.move_right(speed_multiplier)
        else:
            self.enemy.move_left(speed_multiplier)

    def _apply_fighter_physics(self, dt_seconds):
        self.player.apply_physics(dt_seconds, self.STAGE_MIN_X, self.STAGE_MAX_X)
        self.enemy.apply_physics(dt_seconds, self.STAGE_MIN_X, self.STAGE_MAX_X)


    def _update_facing(self):
        if self.player.x <= self.enemy.x:
            self.player.facing = 1
            self.enemy.facing = -1
        else:
            self.player.facing = -1
            self.enemy.facing = 1

    def _sync_animators_to_fighters(self):
        self.player_animator.set_position(self.player.position())
        self.enemy_animator.set_position(self.enemy.position())
        self.player_animator.set_flip(self.player.facing == -1)
        self.enemy_animator.set_flip(self.enemy.facing == -1)

    def _update_movement_animation_states(self):
        self._set_movement_animation(self.player, self.player_animator)
        self._set_movement_animation(self.enemy, self.enemy_animator)

    def _set_movement_animation(self, fighter, animator):
        if animator.state in ("attack", "hit", "death"):
            return

        if fighter.is_dead():
            animator.set_state("death")
        elif not fighter.on_ground:
            animator.set_state("jump")
        elif abs(fighter.velocity_x) > 1:
            animator.set_state("run")
            is_running_backward = (fighter.velocity_x < 0 and fighter.facing == 1) or \
                                  (fighter.velocity_x > 0 and fighter.facing == -1)
            animator.set_reverse_playback(is_running_backward)
        else:
            animator.set_state("idle")
            animator.set_reverse_playback(False)

    def draw(self):
        self.ui.draw(
            self.player,
            self.enemy,
            self.battle_log,
            self.high_score,
            self.enemy.boss_phase,
        )
        self.player_animator.draw(self.screen)
        self.enemy_animator.draw(self.screen)

        if self.game_over:
            self._draw_game_over()

        pygame.display.flip()

    def _draw_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 130))
        self.screen.blit(overlay, (0, 0))

        font = pygame.font.SysFont("arial", 42, bold=True)
        small_font = pygame.font.SysFont("arial", 24)
        result = "DRAW" if self.winner == "Draw" else f"{self.winner.upper()} WINS"
        text = font.render(result, True, (255, 255, 255))
        restart = small_font.render("Press R to restart", True, (230, 230, 230))

        self.screen.blit(text, text.get_rect(center=(self.WIDTH // 2, 300)))
        self.screen.blit(restart, restart.get_rect(center=(self.WIDTH // 2, 355)))
