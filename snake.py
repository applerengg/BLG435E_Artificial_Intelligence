import pygame
from snake_environment import *
from snake_renderer import *
import numpy as np


class CustomSnake(Snake):
    def __init__(self,
                 human=False,
                 mode=SnakeMode.NOTAIL,
                 render=True,
                 game_window_name="Snake Variants - 5000 in 1"):
        super(CustomSnake, self).__init__(human, mode, Renderer(game_window_name) if render else None)

        """
        DEFINE YOUR OBSERVATION SPACE DIMENSIONS HERE FOR EACH MODE.
        JUST CHANGING THE "obs_space_dim" VARIABLE SHOULD BE ENOUGH
        """
        if mode == SnakeMode.NOTAIL:
            obs_space_dim = 5
            self.observation_space = spaces.Box(0, obs_space_dim, shape=(obs_space_dim,))
        elif mode == SnakeMode.CLASSIC:
            obs_space_dim = 5
            self.observation_space = spaces.Box(0, obs_space_dim, shape=(obs_space_dim,))
        elif mode == SnakeMode.TRON:
            obs_space_dim = 5
            self.observation_space = spaces.Box(0, obs_space_dim, shape=(obs_space_dim,))

    def get_state(self):
        """
        Define your state representation here
        :return:
        """
        if self.game_mode == SnakeMode.NOTAIL:
            raise NotImplementedError("Implement your state representation for Snake-NoTail")
        elif self.game_mode == SnakeMode.CLASSIC:
            raise NotImplementedError("Implement your state representation for Snake-Classic")
        elif self.game_mode == SnakeMode.TRON:
            raise NotImplementedError("Implement your state representation for Snake-Tron")
        else:
            raise ModuleNotFoundError("This mode is currently not supported. Please refer to the manual")

    def get_reward(self):
        """
        Define your reward calculations here
        :return:
            A value between (-1, 1)
        """
        if self.game_mode == SnakeMode.NOTAIL:
            raise NotImplementedError("Implement your reward function for Snake-NoTail")
        elif self.game_mode == SnakeMode.CLASSIC:
            raise NotImplementedError("Implement your reward function for Snake-Classic")
        elif self.game_mode == SnakeMode.TRON:
            raise NotImplementedError("Implement your reward function for Snake-Tron")
        else:
            raise ModuleNotFoundError("This mode is currently not supported. Please refer to the manual")
