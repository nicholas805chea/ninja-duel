"""User interface rendering for Ninja Duel."""

from pathlib import Path

import pygame


class UI:
    """Draws HP bars, score, high score, action buttons, and battle log."""

    ACTION_LABELS = {
        "Attack": "8 Attack",
        "Defend": "9 Defend",
        "Heal": "0 Heal",
    }

    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont("arial", 22)
        self.small_font = pygame.font.SysFont("arial", 18)
        self.title_font = pygame.font.SysFont("arial", 30, bold=True)
        self.background = self._load_background()
        # Defined here (not at class level) so pygame is already initialised.
        self.ACTION_BUTTONS = {
            "Attack": pygame.Rect(80, 635, 150, 54),
            "Defend": pygame.Rect(250, 635, 150, 54),
            "Heal": pygame.Rect(420, 635, 150, 54),
        }

    def _load_background(self):
        """Load the dojo JPG once and scale it to the game window."""
        background_path = Path("dojo background.jpg")
        if not background_path.exists():
            return None

        image = pygame.image.load(str(background_path)).convert()
        return pygame.transform.smoothscale(image, self.screen.get_size())

    def get_clicked_action(self, mouse_pos):
        """Action boxes are labels only; combat is controlled by number keys."""
        return None

    def draw(self, player, enemy, battle_log, high_score, boss_phase):
        self._draw_background()
        self._draw_hp_bar(60, 40, "Player", player.hp, player.max_hp, (42, 178, 92))
        self._draw_hp_bar(620, 40, "Enemy", enemy.hp, enemy.max_hp, (200, 65, 65))
        self._draw_score(player.score, high_score)
        self._draw_buttons()
        self._draw_battle_log(battle_log)

        if boss_phase:
            self._draw_boss_phase_banner()

    def _draw_background(self):
        if self.background:
            self.screen.blit(self.background, (0, 0))
        else:
            self.screen.fill((26, 29, 36))

        lower_overlay = pygame.Surface((1000, 210), pygame.SRCALPHA)
        lower_overlay.fill((18, 20, 25, 150))
        self.screen.blit(lower_overlay, (0, 510))
        pygame.draw.line(self.screen, (95, 92, 84), (0, 510), (1000, 510), 4)

    def _draw_hp_bar(self, x, y, label, hp, max_hp, color):
        width = 320
        height = 28
        ratio = hp / max_hp

        label_surface = self.font.render(f"{label} HP: {hp}/{max_hp}", True, (245, 245, 245))
        self.screen.blit(label_surface, (x, y - 30))

        pygame.draw.rect(self.screen, (18, 20, 25), (x, y, width, height), border_radius=4)
        pygame.draw.rect(
            self.screen,
            color,
            (x, y, int(width * ratio), height),
            border_radius=4,
        )
        pygame.draw.rect(self.screen, (230, 230, 230), (x, y, width, height), 2, 4)

    def _draw_score(self, score, high_score):
        score_text = self.font.render(f"Score: {score}", True, (245, 245, 245))
        high_score_text = self.font.render(
            f"High Score: {high_score}", True, (245, 245, 245)
        )
        self.screen.blit(score_text, (60, 95))
        self.screen.blit(high_score_text, (60, 125))

    def _draw_buttons(self):
        colors = {
            "Attack": (176, 52, 52),
            "Defend": (52, 98, 176),
            "Heal": (52, 145, 83),
        }

        for action, rect in self.ACTION_BUTTONS.items():
            color = colors[action]
            pygame.draw.rect(self.screen, color, rect, border_radius=6)
            pygame.draw.rect(self.screen, (235, 235, 235), rect, 2, 6)

            text = self.font.render(self.ACTION_LABELS[action], True, (255, 255, 255))
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)

    def _draw_battle_log(self, battle_log):
        panel = pygame.Rect(620, 565, 330, 125)
        pygame.draw.rect(self.screen, (16, 18, 23), panel, border_radius=6)
        pygame.draw.rect(self.screen, (85, 88, 98), panel, 2, 6)

        title = self.small_font.render("Battle Log", True, (245, 245, 245))
        self.screen.blit(title, (panel.x + 12, panel.y + 8))

        for index, message in enumerate(battle_log[-5:]):
            line = self.small_font.render(message, True, (220, 220, 220))
            self.screen.blit(line, (panel.x + 12, panel.y + 34 + index * 18))

    def _draw_boss_phase_banner(self):
        banner = self.title_font.render("BOSS PHASE ACTIVATED", True, (255, 210, 70))
        rect = banner.get_rect(center=(500, 145))
        pygame.draw.rect(
            self.screen,
            (88, 35, 35),
            rect.inflate(34, 18),
            border_radius=6,
        )
        self.screen.blit(banner, rect)
