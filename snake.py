import pygame
from snake_environment import *
from snake_renderer import *
import numpy as np
import config


class CustomSnake(Snake):
    def __init__(self,
                 human=False,
                 mode=SnakeMode.NOTAIL,
                 render=True,
                 game_window_name="Snake Variants - 5000 in 1"):
        super(CustomSnake, self).__init__(human, mode, Renderer(game_window_name) if render else None)

        self.steps_since_apple = 0 # counter for preventing snake from moving in circles
        self.apple_dist = 0
        self.apple_dist_prev = 0

        """
        DEFINE YOUR OBSERVATION SPACE DIMENSIONS HERE FOR EACH MODE.
        JUST CHANGING THE "obs_space_dim" VARIABLE SHOULD BE ENOUGH
        """
        if mode == SnakeMode.NOTAIL:
            obs_space_dim = 11
            self.observation_space = spaces.Box(0, obs_space_dim, shape=(obs_space_dim,))
        elif mode == SnakeMode.CLASSIC:
            obs_space_dim = 11
            self.observation_space = spaces.Box(0, obs_space_dim, shape=(obs_space_dim,))
        elif mode == SnakeMode.TRON:
            obs_space_dim = 11
            self.observation_space = spaces.Box(0, obs_space_dim, shape=(obs_space_dim,))
            # obs_space_dim = GRID_DIMS + 2
            # self.observation_space = spaces.Box(0, obs_space_dim, shape=(obs_space_dim,obs_space_dim))

    def get_state(self):
        """
        Define your state representation here
        :return:
        """
        # self.state_space = []
        # self.state_space.append(self.snake_pos)
        snake_row, snake_col = self.snake_pos
        snake_dir = convert_direction_to_action(self.snake_direction)
        # if self.game_mode in [SnakeMode.NOTAIL, SnakeMode.CLASSIC]: # do not care apple in TRON mode
        if self.game_mode in [SnakeMode.NOTAIL, SnakeMode.CLASSIC, SnakeMode.TRON]:
            apple_row, apple_col = self.apple_pos
            self.apple_dist_prev = self.apple_dist
            self.apple_dist = self._manhattan_distance(snake_row, snake_col, apple_row, apple_col)
            pos_diff = (apple_row - snake_row, apple_col - snake_col)
            apple_dir = convert_vector_to_action(pos_diff)  # 0,1,2,3 for directions or -1
        
        wall_locations = [0, 0, 0, 0] # will be set to 1 for up, left, down, right directions
        wall_locations[0] =  snake_row == 1             # up
        wall_locations[1] =  snake_col == 1             # left
        wall_locations[2] =  snake_row == GRID_DIMS     # down
        wall_locations[3] =  snake_col == GRID_DIMS     # right
            

        if self.game_mode == SnakeMode.NOTAIL: # use only wall locations (no body)
            self.state_space = [snake_row, snake_col, apple_row, apple_col, self.apple_dist, 
                                snake_dir, apple_dir, *wall_locations]
        elif self.game_mode == SnakeMode.CLASSIC: # use obstacle locations (wall+body)
            body_locations = [0, 0, 0, 0] # will be set to 1 for up, left, down, right directions
            body_locations[0] = self.level_matrix[snake_row - 1][snake_col] == SNAKE_BODY # U
            body_locations[1] = self.level_matrix[snake_row][snake_col - 1] == SNAKE_BODY # L
            body_locations[2] = self.level_matrix[snake_row + 1][snake_col] == SNAKE_BODY # D
            body_locations[3] = self.level_matrix[snake_row][snake_col + 1] == SNAKE_BODY # R
            obstacle_locations = np.logical_or(wall_locations, body_locations)
            self.state_space = [snake_row, snake_col, apple_row, apple_col, self.apple_dist,
                                snake_dir, apple_dir, *obstacle_locations]

        elif self.game_mode == SnakeMode.TRON: # use obstacle locations (wall+body), do not care apple
            body_locations = [0, 0, 0, 0] # will be set to 1 for up, left, down, right directions
            body_locations[0] = self.level_matrix[snake_row - 1][snake_col] == SNAKE_BODY # U
            body_locations[1] = self.level_matrix[snake_row][snake_col - 1] == SNAKE_BODY # L
            body_locations[2] = self.level_matrix[snake_row + 1][snake_col] == SNAKE_BODY # D
            body_locations[3] = self.level_matrix[snake_row][snake_col + 1] == SNAKE_BODY # R
            obstacle_locations = np.logical_or(wall_locations, body_locations)
            # snake_dirs = [0, 0, 0, 0]
            # snake_dirs[snake_dir] = 1
            # self.state_space = [
            #     snake_row, snake_col,
            #     snake_dir, *obstacle_locations
            #     ]
            self.state_space = [snake_row, snake_col, apple_row, apple_col, self.apple_dist,
                                snake_dir, apple_dir, *obstacle_locations]
            # obstacles = (SNAKE_BODY, WALL)
            # self.state_space = np.array(
            #     [[1 if val in obstacles else 0 for val in row] for row in self.level_matrix]
            # )
            # self.state_space = [list(map(ord, row)) for row in self.level_matrix]

        else:
            raise ModuleNotFoundError("This mode is currently not supported. Please refer to the manual")
        
        # print(self.state_space)
        return self.state_space


    def get_reward(self):
        """
        Define your reward calculations here
        :return:
            A value between (-1, 1) --> will be normalized
        """
        apple_reward = 20
        if self.game_mode in [SnakeMode.NOTAIL, SnakeMode.CLASSIC]: # do not care apple in TRON mode
            apple_dist_decreased = self.apple_dist < self.apple_dist_prev
            apple_dist_increased = self.apple_dist > self.apple_dist_prev
            
            if self.hit_the_wall: # snake hit the wall or its body
                self.reward = -100      # dying penalty
            # elif self.steps_since_apple > apple_reward:
            #     self.reward = 1        # prevent snake from moving in circles
            elif self.ate_apple:
                self.reward = apple_reward  # eating apple reward
            elif apple_dist_decreased:
                self.reward = 1
            elif apple_dist_increased:
                self.reward = -1
            else:
                self.reward = 0         # surviving reward
            
            self.total_reward += self.reward
        
        
        elif self.game_mode == SnakeMode.TRON:
            apple_dist_decreased = self.apple_dist < self.apple_dist_prev
            apple_dist_increased = self.apple_dist > self.apple_dist_prev
            
            if self.hit_the_wall: # snake hit the wall or its body
                self.reward = -100      # dying penalty
            elif self.ate_apple:
                self.reward = apple_reward  # eating apple reward
            # elif apple_dist_decreased:
            #     self.reward = 1
            # elif apple_dist_increased:
            #     self.reward = -1
            else:
                self.reward = 10         # surviving reward
            
            self.total_reward += self.reward

        else:
            raise ModuleNotFoundError("This mode is currently not supported. Please refer to the manual")

        return self.reward



    def _manhattan_distance(self, r1, c1, r2, c2):
        return np.abs(r1 - r2) + np.abs(c1 - c2)



    def asd():
        """
        xx xx uu xx xx
        xx ul u_ ur xx
        ll l_ __ r_ rr
        xx dl d_ dr xx
        xx xx dd xx xx

        """
        pass
