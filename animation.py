"""Sprite sheet animation utilities."""

from pathlib import Path

import pygame


class SpriteAnimation:
    """Loads a horizontal sprite sheet and plays it frame by frame."""

    def __init__(self, image_path, scale=2.8, frame_duration=90, flip=False):
        self.image_path = Path(image_path)
        self.scale = scale
        self.frame_duration = frame_duration
        self.default_flip = flip
        self.frames = []
        self.current_frame = 0
        self.elapsed_ms = 0
        self.finished = False
        self._load_frames()

    def _load_frames(self):
        if not self.image_path.exists():
            self.frames = [self._fallback_surface()]
            return

        sheet = pygame.image.load(str(self.image_path)).convert_alpha()
        sheet_width, sheet_height = sheet.get_size()
        frame_width = sheet_height
        frame_count = max(1, sheet_width // frame_width)

        for index in range(frame_count):
            frame = sheet.subsurface(
                pygame.Rect(index * frame_width, 0, frame_width, sheet_height)
            )
            scaled_size = (
                int(frame_width * self.scale),
                int(sheet_height * self.scale),
            )
            frame = pygame.transform.scale(frame, scaled_size)
            self.frames.append(frame)

    def _fallback_surface(self):
        surface = pygame.Surface((120, 120), pygame.SRCALPHA)
        pygame.draw.rect(surface, (70, 70, 80), surface.get_rect(), border_radius=10)
        pygame.draw.rect(surface, (230, 230, 240), surface.get_rect(), 3, 10)
        return surface

    def reset(self):
        self.current_frame = 0
        self.elapsed_ms = 0
        self.finished = False

    def update(self, dt_ms, loop=True, reverse=False):
        if len(self.frames) <= 1 or self.finished:
            return

        self.elapsed_ms += dt_ms
        if self.elapsed_ms < self.frame_duration:
            return

        self.elapsed_ms = 0

        if reverse:
            self.current_frame -= 1
            if self.current_frame < 0:
                if loop:
                    self.current_frame = len(self.frames) - 1
                else:
                    self.current_frame = 0
                    self.finished = True
        else:
            self.current_frame += 1
            if self.current_frame >= len(self.frames):
                if loop:
                    self.current_frame = 0
                else:
                    self.current_frame = len(self.frames) - 1
                    self.finished = True

    def get_frame(self, flip=None):
        frame = self.frames[self.current_frame]
        should_flip = self.default_flip if flip is None else flip
        if should_flip:
            return pygame.transform.flip(frame, True, False)
        return frame


class FighterAnimator:
    """Animation state machine for combat and real-time movement."""

    STATE_TO_FILE = {
        "idle": "Idle.png",
        "run": "Run.png",
        "jump": "Jump.png",
        "attack": "Attack1.png",
        "hit": "Take Hit.png",
        "death": "Death.png",
    }

    NON_LOOPING_STATES = {"attack", "hit", "death"}

    def __init__(self, sprite_dir, position, flip=False):
        self.sprite_dir = Path(sprite_dir)
        self.position = position
        self.flip = flip
        self.state = "idle"
        self.reverse_playback = False
        self.animations = {
            state: SpriteAnimation(self.sprite_dir / filename, flip=flip)
            for state, filename in self.STATE_TO_FILE.items()
        }

    def set_state(self, state):
        if state not in self.animations:
            state = "idle"
        if state == self.state:
            return
        self.state = state
        self.animations[self.state].reset()

    def set_position(self, position):
        self.position = position

    def set_flip(self, flip):
        self.flip = flip

    def set_reverse_playback(self, reverse):
        self.reverse_playback = reverse

    def update(self, dt_ms):
        animation = self.animations[self.state]
        loop = self.state not in self.NON_LOOPING_STATES
        reverse = self.reverse_playback if self.state == "run" else False
        animation.update(dt_ms, loop=loop, reverse=reverse)

        if animation.finished and self.state != "death":
            self.set_state("idle")

    def draw(self, surface):
        frame = self.animations[self.state].get_frame(flip=self.flip)
        rect = frame.get_rect(midbottom=self.position)
        surface.blit(frame, rect)
