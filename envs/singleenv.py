from babyai.levels.levels import RoomGridLevel
from babyai.levels.roomgrid import Ball, Key, Box
from babyai.levels.instrs import Instr, Object
from gym_minigrid.minigrid import COLOR_NAMES
from gym_minigrid.envs import DIR_TO_VEC

class SEnv(RoomGridLevel):
    def __init__(self, num_doors=4, locked_proba=0.5, inbox_proba=0.5, blocked_proba=0.5, seed=None):
        self.num_doors = num_doors
        self.locked_proba = locked_proba
        self.inbox_proba = inbox_proba
        self.blocked_proba = blocked_proba

        room_size = 6
        super().__init__(
            room_size=room_size,
            max_steps=4*(1+self.num_doors)*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        door_colors = self._rand_subset(COLOR_NAMES, self.num_doors)
        obj_colors = self._rand_subset(COLOR_NAMES[1:], self.num_doors)
        
        for i in range(self.num_doors):
            vec = DIR_TO_VEC[i]

            # Add a box to room in dir i
            obj_color = obj_colors[i]
            self.add_object(1+vec[0], 1+vec[1], kind="ball", color=obj_color)

            # Add a door in dir i
            door_color = door_colors[i]
            locked = self._rand_float(0, 1) < self.locked_proba
            _, door_pos = self.add_door(1, 1, door_idx=i, color=door_color, locked=locked)

            # If the door is locked, a key is placed in the center room
            if locked:
                obj = Key(door_color)
                inbox = self._rand_float(0, 1) < self.inbox_proba
                if inbox:
                    obj = Box(self._rand_color(), obj)
                self.place_in_room(1, 1, obj)

            # If the door must be blocked, a ball is placed in front of the door
            blocked = self._rand_float(0, 1) < self.blocked_proba
            if blocked:
                self.grid.set(door_pos[0]-vec[0], door_pos[1]-vec[1], Ball(COLOR_NAMES[0]))

        self.place_agent(1, 1)

        obj_color = self._rand_elem(obj_colors)
        self.instrs = [Instr(action="pickup", object=Object("ball", obj_color))]

def get_several(getter, seed, num_procs):
    return [getter(seed + shift) for shift in range(num_procs)]

def get_several_SEnv_D1LnInBn(seed, num_procs):
    return get_several(get_SEnv_D1LnInBn, seed, num_procs)

def get_SEnv_D1LnInBn(seed):
    return SEnv(1, 0, 0, 0, seed)

def get_several_SEnv_D1LaInBn(seed, num_procs):
    return get_several(get_SEnv_D1LaInBn, seed, num_procs)

def get_SEnv_D1LaInBn(seed):
    return SEnv(1, 1, 0, 0, seed)
        
def get_several_SEnv_D1LnInBa(seed, num_procs):
    return get_several(get_SEnv_D1LnInBa, seed, num_procs)

def get_SEnv_D1LnInBa(seed):
    return SEnv(1, 0, 0, 1, seed)
        
def get_several_SEnv_D1LaInBa(seed, num_procs):
    return get_several(get_SEnv_D1LaInBa, seed, num_procs)

def get_SEnv_D1LaInBa(seed):
    return SEnv(1, 1, 0, 1, seed)
        
def get_several_SEnv_D1LaIaBn(seed, num_procs):
    return get_several(get_SEnv_D1LaIaBn, seed, num_procs)

def get_SEnv_D1LaIaBn(seed):
    return SEnv(1, 1, 1, 0, seed)
        
def get_several_SEnv_D2LnInBn(seed, num_procs):
    return get_several(get_SEnv_D2LnInBn, seed, num_procs)

def get_SEnv_D2LnInBn(seed):
    return SEnv(2, 0, 0, 0, seed)
        
def get_several_SEnv_D4LnInBn(seed, num_procs):
    return get_several(get_SEnv_D4LnInBn, seed, num_procs)

def get_SEnv_D4LnInBn(seed):
    return SEnv(4, 0, 0, 0, seed)

def get_several_SEnv_D4LuIuBu(seed, num_procs):
    return get_several(get_SEnv_D4LuIuBu, seed, num_procs)

def get_SEnv_D4LuIuBu(seed):
    return SEnv(4, 0.5, 0.5, 0.5, seed)