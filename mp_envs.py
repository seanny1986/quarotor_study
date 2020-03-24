import gym
import gym_aero

def make(name):
    if name=="Hover-v0": return make_hover()
    if name=="PIDHover-v0": return make_pid_hover()
    if name=="RandomWaypointFH-v0": return make_randomwaypointfh()
    if name=="PIDRandomWaypointFH-v0": return make_pid_randomwaypointfh()
    if name=="RandomWaypointNH-v0": return make_randomwaypointnh()
    if name=="PIDRandomWaypointNH-v0": return make_pid_randomwaypointnh()
    if name=="Land-v0": return make_land()
    if name=="PIDLand-v0": return make_pid_land()

def make_hover():
    def _thunk():
        env = gym.make("Hover-v0")
        return env
    return _thunk

def make_pid_hover():
    def _thunk():
        env = gym.make("Hover-v0")
        return env
    return _thunk

def make_randomwaypointfh():
    def _thunk():
        env = gym.make("RandomWaypointFH-v0")
        return env
    return _thunk

def make_pid_randomwaypointfh():
    def _thunk():
        env = gym.make("PIDRandomWaypointFH-v0")
        return env
    return _thunk

def make_randomwaypointnh():
    def _thunk():
        env = gym.make("RandomWaypointNH-v0")
        return env
    return _thunk

def make_pid_randomwaypointnh():
    def _thunk():
        env = gym.make("PIDRandomWaypointNH-v0")
        return env
    return _thunk

def make_land():
    def _thunk():
        env = gym.make("Land-v0")
        return env
    return _thunk

def make_pid_land():
    def _thunk():
        env = gym.make("PIDLand-v0")
        return env
    return _thunk

def make_half_cheetah():
    def _thunk():
        env = gym.make("HalfCheetah-v1")
        return env
    return _thunk