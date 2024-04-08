from enum import IntEnum
from typing import Tuple, Optional, List
from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np


def register_env() -> None:
    """Register custom gym environment so that we can use `gym.make()`

    This will register the custom environment with Gymnasium. We are using this for compatibility and reproducability.  
    """
    register(id="Ball-Balancing-Table-v0", entry_point="env:BallBalancingTable")


class Action(IntEnum):
    """Represents an action for the Ball Balancing Table at a single time step to control the 2 servomotors. 
    We will assume that each servomotor can move in either direction (towards the table or away from it) and either apply force, retract and lessen the force, or do nothing. 
    The applied force will be discretized for each time step to add or remove 'x' N (or do nothing)"""
    # X_X --> Action for Motor 1 _ Action for Motor 2
    # N --> Nothing
    # U --> Up
    # D --> Down
    N_N = 0 
    N_U = 1
    N_D = 2
    
    U_N = 3
    U_U = 4
    U_D = 5
    
    D_N = 6
    D_U = 7
    D_D = 8


def actions_to_forces(action: Action, force_step: float) -> Tuple[int, int]:
    """
    Helper function to map action to changes in servomotor forces
    Args:
        action (Action): taken action
        force_step (float): The step applied to the servomotors
    Returns:
        dxdy (Tuple[int, int]): Change in force
    """
    mapping = {
        Action.N_N: (0, 0),
        Action.N_U: (0, force_step),
        Action.N_D: (0, -1*force_step),
        
        Action.U_N: (force_step, 0),
        Action.U_U: (force_step, force_step),
        Action.U_D: (force_step, -1*force_step),
        
        Action.D_N: (-1*force_step, 0),
        Action.D_U: (-1*force_step, force_step),
        Action.D_D: (-1*force_step, -1*force_step),
        
    }
    return mapping[action]


class BallBalancingTable(Env):
    """Ball Balancing gym environment.

    This will serve as our model for the ball balancing table. The init method can take in boolean flags to add gaussian noise to the system
    
    __init__ Params:
    sensor_noise: A boolean to determine whether to add sensor noise
    sensor_std: Standard deviation of the normal distribution to represent sensor noise
    sensor_sensitivity: The decimal places representing the sensitivity of the table's pressure sensor (readings are in m)
    ball_mass: the mass of the metal ball (Kg) placed on the table (please note that the radius of the ball is unneeded as it cancels out --> check the system dynamics section of the project)
    table_mass: mass of the glass surface/table (Kg) to determine the inertia of the table movement
    table_length: length of one side of the square table (m)
    force_step: The magnitude of force each servomotor can apply at a given timestep, this can be changed along with dt to represent the actual servomotors' performance if the real system specifications are provided
    dt: the time step to be used in this environment
    angle_limit: The maximum angle the table can produce (models a physical limitation)
    force_limit: The maximum force the servomotors can produce
    """

    def __init__(self, sensor_noise: bool=False, sensor_std: float = 0.5, sensor_sensitivity:float=2, ball_mass: float=1, table_mass:float=5, table_length: float=0.3, dt:float=0.1, force_step: float=1, angle_limit: float=30, force_limit: float=10, max_damping:float=3) -> None:
        # Gymnasium Params
        self.np_random = seeding.np_random()
        self.action_space = spaces.Discrete(len(Action))
        
        # Input Params
        self.ball_mass = ball_mass
        self.table_mass = table_mass
        self.dt = dt
        self.force_step = force_step
        self.table_length = table_length
        self.r = table_length/2
        self.sensor_noise = sensor_noise
        self.sensor_std = sensor_std
        self.sensor_sensitivity = sensor_sensitivity
        self.angle_limit = angle_limit
        self.force_limit = force_limit
        self.max_damping = max_damping # The max resistive forces acting against the servomotors in each direction (natural damping from the table). In reality, this will require good knowledge of the system or (more efficiently) will be calculated from experiment data where you deduce the system damping based on angle changes when you apply certain forces. The damping force will generally equal 30% of the force output up to the max damping.
        
        # Variables for Table Dynamics
        self.table_inertia = (4.0/12.0)*self.table_mass*(self.r**2)
        # Gravity
        self.g = 9.81 
        # The current forces acting on the table (Motor 1 Force, Motor 2 Force)
        self.force1 = 0.0
        self.force2 = 0.0 
        # Angular accelaration of the table about x and y directions
        self.theta_x_acc = 0
        self.theta_y_acc = 0
        # Angular velocity of the table about x and y directions
        self.theta_x_vel = 0
        self.theta_y_vel = 0
        # Angles of the table about x and y directions
        self.theta_x = 0
        self.theta_y = 0
        # Ball accelarations in x and y direction
        self.ball_acc_x = 0.0
        self.ball_acc_y = 0.0
        # Ball Velocities in x and y direction
        self.ball_vel_x = 0.0
        self.ball_vel_y = 0.0
        # Ball position
        self.x = 0.0
        self.y = 0.0
        
        # Game state 
        self.ball_pos = (self.x,self.y)
        


    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the the ball position)
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this simulation.
        """
        # Setting the rewards for the game
        reward_standard = -1 # Standard reward for each time step
        reward_loss = -100 # Reward for losing (ball falling off)
        reward_win = 100
        
        # Gets the change is servomotor forces from the action and calculates new forces
        force_changes = actions_to_forces(action, self.force_step)
        new_f1 = self.force1 + force_changes[0]
        new_f2 = self.force2 + force_changes[1]
        
        # Clips the forces based on servomotor limits
        self.force1 = np.clip(new_f1, -1* self.force_limit, self.force_limit)
        self.force2 = np.clip(new_f2, -1* self.force_limit, self.force_limit)
        
        # Updates the theta based on the new forces
        self.update_theta()
        
        # Updates the ball dynamics (position, velocity, accelaration)
        self.update_ball()
        
        # "Measures/Senses" the ball position and adds noise if the boolean flag is True
        # The true position will not be returned to the agent, but instead the sensed x and y positions only
        self.sense_ball()
        
        # If the ball falls off the table, return an ended game with negative rewards
        if not self.in_bounds(self.x, self.r) or not self.in_bounds(self.y, self.r):
            return self.ball_pos, reward_loss, True, {}
        
        # If the ball reaches steady state
        if self.won_game():
            return self.ball_pos, reward_win, True, {}
        
        return self.ball_pos, reward_standard, False, {}
    
    
    def update_ball(self):
        """Updates the ball accelaration, velocity, and position based on system dynamics
        """
        # Using equations of motion to determine the current ball accelarations
        self.ball_acc_x = (2.0/3.0)*self.g*np.sin(self.theta_x)
        self.ball_acc_y = (2.0/3.0)*self.g*np.sin(self.theta_y)
        
        # Calculates ball velocity based on accelaration
        self.ball_vel_x = self.ball_vel_x + self.ball_acc_x*self.dt
        self.ball_vel_y = self.ball_vel_y + self.ball_acc_y*self.dt
        
        self.x = self.x + self.ball_vel_x*self.dt
        self.y = self.y + self.ball_vel_y*self.dt
        
    
    def update_theta(self):
        """Updates the table angles based on the current forces on the table, based on the table dynamics equations. 
        However, the simulation also checks whether the new angle is producable given the table angular limits before updating
        if it cannot be done, we assume the table angles remain the same and the force does not affect it (represents physical limits)
        """
        # Calculating the damping force of the system, usually =30% of the given force but limited at self.max_damping
        damp1 = np.clip(self.force1*0.3, -1*self.max_damping, self.max_damping)
        damp2 = np.clip(self.force2*0.3, -1*self.max_damping, self.max_damping)
        
        # Using the equations of motion to determine the angular accelaration of the table based on the current forces
        x_acc = (self.r*(self.force1 - damp1) + self.ball_mass*self.g*self.x)/self.table_inertia
        y_acc = (self.r*(self.force2 - damp2) + self.ball_mass*self.g*self.y)/self.table_inertia
        
        # Calculates the new angular velocity based on the accelarations
        x_vel = self.theta_x_vel + x_acc*self.dt
        y_vel = self.theta_y_vel + y_acc*self.dt  
        
        # Calculates the new angles of the table around the x and y axes
        x = self.theta_x + x_vel*self.dt
        y = self.theta_y + y_vel*self.dt
        
        # Sets the new theta_x variables if within limits
        if self.in_bounds(x, self.angle_limit):
            self.theta_x_acc = x_acc
            self.theta_x_vel = x_vel
            self.x = x
        
        # Sets the new theta_y variables if within limits
        if self.in_bounds(y, self.angle_limit):
            self.theta_y_acc = y_acc
            self.theta_y_vel = y_vel
            self.y = y
    
    def sense_ball(self):
        """ Acts as the pressure sensor in the glass surface, has a certain sensitivity and can have gaussian noise
        """ 
        self.ball_pos = (round(self.x, self.sensor_sensitivity), round(self.y, self.sensor_sensitivity))
        
        # If we want to add noise to the sensor
        if self.sensor_noise:
            x = np.random.normal(self.x, self.sensor_std)
            y = np.random.normal(self.y, self.sensor_std)
            self.ball_pos = (round(x, self.sensor_sensitivity), round(y, self.sensor_sensitivity))
            
    def won_game(self) -> bool:
        """Returns whether the game has been won or not (reached steady state)
        Win conditions are whether ball position is within 2cm of the origin (0,0) and has a relatively low velocity and accelaration
        """
        b1 = self.in_bounds(self.x, 0.02) and self.in_bounds(self.y, 0.02)
        b2 = self.in_bounds(self.ball_vel_x, 0.001) and self.in_bounds(self.ball_vel_y, 0.001)
        b3 = self.in_bounds(self.ball_acc_x, 0.001) and self.in_bounds(self.ball_acc_y, 0.001)
        return b1 and b2 and b3

    def in_bounds(self, x:float, limit:float) -> bool:
        """Helper method that checks if the given x value is within +/- the given limit

        Args:
            x (float): arg
            x_limits (float): limit

        Returns:
            bool: Whether the ball is in bounds
        """
    
        return -1*limit <= x <= limit     
            
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Fix seed of environment
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self, starting_ball_pos: Tuple[float, float]=(0,0), starting_ball_vel: Tuple[float, float]=(0,0)) -> Tuple[int, int]:
        """Resets agent to the starting position. Can also take in a ball position and velocity to start at.

        Returns:
            observation (Tuple[int,int]): returns the initial observation
        """
        self.ball_pos = starting_ball_pos
        self.ball_vel = starting_ball_vel
        return self.ball_pos
    

