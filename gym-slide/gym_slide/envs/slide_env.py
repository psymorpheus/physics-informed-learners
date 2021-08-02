import math

import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

class SlideEnv(gym.Env):
    """
    Description:
        The agent (a car) is started at the bottom of a valley. For any given
        state the agent may choose to accelerate to the left, right or cease
        any acceleration.
    Source:
        The environment appeared first in Andrew Moore's PhD Thesis (1990).
    Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07
    Actions:
        Type: Discrete(3)
        Num    Action
        0      Accelerate to the Left
        1      Don't accelerate
        2      Accelerate to the Right
        Note: This does not affect the amount of velocity affected by the
        gravitational pull acting on the car.
    Reward:
         Reward of 0 is awarded if the agent reached the flag (position = 0.5)
         on top of the mountain.
         Reward of -1 is awarded if the position of the agent is less than 0.5.
    Starting State:
         The position of the car is assigned a uniform random value in
         [-0.6 , -0.4].
         The starting velocity of the car is always assigned to 0.
    Episode Termination:
         The car position is more than 0.5
         Episode length is greater than 200
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, goal_velocity=0):
        self.min_position = -2.0
        self.max_position = 2.0
        self.max_speed = 2
        self.goal_position = 1.5
        self.goal_velocity = goal_velocity

        # self.force = 0.001 # Some function of time decided later
        self.gravity = -9.8
        self.mass = 20.0 # mass in kg

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.viewer = None

        # self.action_space = spaces.Box( np.array([-1,0,0]), np.array([+1,+1,+1]))
        self.min_action = -1000.0
        self.max_action = 1000.0
        self.action_space = spaces.Box( np.array([self.min_action, self.min_action]), np.array([self.max_action, self.max_action]))     # Only force is given as input, ForceX, ForceY
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.time_step = 0.01       # Time between each episode (in seconds)
        self.mu = 0.5               # Coefficient of friction
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        position, velocity = self.state
        force = {'x':min(max(action[0], self.min_action), self.max_action), 'y':min(max(action[1], self.min_action), self.max_action)}

        theta = self._theta(position)
        force_along = force['x'] * math.cos(theta) + force['y'] * math.sin(theta)
        force_perp = -1 * force['x'] * math.sin(theta) + force['y'] * math.cos(theta)     # Out of ground positive
        normal = force_perp - self.mass * self.gravity * math.cos(theta)

        force_along += math.sin(theta) * (self.gravity) * self.mass

        # To take care of friction
        if velocity>0:
          force_along -= self.mu * normal
        elif velocity<0:
          force_along += self.mu * normal
        else:
          if abs(force_along)<=abs(self.mu * normal):
            force_along = 0
          else:
            if force_along>0: force_along -= abs(self.mu * normal)
            else: force_along += abs(self.mu * normal)

        # Path dependent (kinematics)
        velocity += (force_along/self.mass) * self.time_step     # Component of gravity parallel to surface, v=u+at
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)     # Bounds it within this interval
        
        # Time has been taken as 1
        position += velocity * self.time_step * math.cos(theta)
        position = np.clip(position, self.min_position, self.max_position)

        # Can't go beyond edge
        if position == self.min_position and velocity < 0:
            velocity = 0

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        if normal<0:
          reward = -1000.0
        else:
          reward = -1.0

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    def _height(self, xs):
        # This is y as a function of x which tells our path
        # Path dependent
        # return np.sin(3 * xs) * 0.45 + 0.55
        return (xs*xs) * 0.45 + 0.2

    def _theta(self, xs):
        return math.atan(2 * xs * 0.45)

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)     # x-axis defined in 100 intervals
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)      # To add elements to the viewer

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(0.5, 0.5, 0.5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(0.8, 0.8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos - self.min_position) * scale, self._height(pos) * scale
        )
        # Path dependent (diff of y wrt x)
        self.cartrans.set_rotation(self._theta(pos))

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def get_keys_to_action(self):
        # Control with left and right arrow keys.
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None