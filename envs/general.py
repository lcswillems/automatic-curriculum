from gym.envs.registration import register
from babyai.levels.levels import RoomGridLevel
from babyai.levels.roomgrid import Key, Box, Ball
from gym_minigrid.envs import DIR_TO_VEC
from gym_minigrid.minigrid import COLOR_NAMES
from babyai.levels.instrs import Instr, Object

class BaseEnv(RoomGridLevel):
    def __init__(self, *args, **kwargs):
        kwargs["room_size"] = 6
        if "num_rooms_visited" in kwargs.keys():
            kwargs["max_steps"] = 4*kwargs["num_rooms_visited"]*kwargs["room_size"]**2
            del kwargs["num_rooms_visited"]

        super().__init__(*args, **kwargs)

    def gen_mission(self):
        self.door_colors = self._rand_subset(COLOR_NAMES, len(COLOR_NAMES))
        self.ball_to_find_color = COLOR_NAMES[0]
        self.blocking_ball_color = COLOR_NAMES[1]
        self.box_color = COLOR_NAMES[2]

        self.instrs = [Instr(action="pickup", object=Object("ball", self.ball_to_find_color))]

    def add_door(self, *args, **kwargs):
        if "in_box" in kwargs.keys():
            in_box = Box(self.box_color) if kwargs["in_box"] else None
            del kwargs["in_box"]
        else:
            in_box = None
        if "blocked" in kwargs.keys():
            blocked_by = Ball(self.blocking_ball_color) if kwargs["blocked"] else None
            del kwargs["blocked"]
        else:
            blocked_by = None

        door, door_pos = super().add_door(*args, **kwargs)

        if blocked_by:
            assert "door_idx" in kwargs.keys() and kwargs["door_idx"] is not None
            idx = kwargs["door_idx"]
            vec = DIR_TO_VEC[idx]
            self.grid.set(door_pos[0]-vec[0], door_pos[1]-vec[1], blocked_by)
        if "locked" in kwargs.keys() and kwargs["locked"]:
            obj = Key(door.color)
            if in_box:
                in_box.contains = obj
                obj = in_box
            self.place_in_room(*args, obj)

        return door, door_pos

class SCEnv(BaseEnv):
    def __init__(self, agent_room, in_box, blocked, num_quarters, num_rooms_visited, seed=None):
        self.agent_room = agent_room
        self.in_box = in_box
        self.blocked = blocked
        self.num_quarters = num_quarters

        super().__init__(
            num_rooms_visited=num_rooms_visited
        )

    def gen_mission(self):
        super().gen_mission()

        for i in range(self.num_quarters):
            middle_room = (1, 1)
            vec = DIR_TO_VEC[i]
            side_room = (middle_room[0] + vec[0], middle_room[1] + vec[1])

            self.add_door(*middle_room, door_idx=i, color=self.door_colors[i], locked=False)

            for k in [-1, 1]:
                self.add_door(*side_room, locked=True,
                              door_idx=(i+k)%4,
                              color=self.door_colors[(i+k)%len(self.door_colors)],
                              in_box=self.in_box,
                              blocked=self.blocked)

        possible_ball_rooms = [(2, 0), (2, 2), (0, 2), (0, 0)][:self.num_quarters]
        ball_room = self._rand_elem(possible_ball_rooms)
        self.add_object(*ball_room, "ball", color=self.ball_to_find_color)
        self.place_agent(*self.agent_room)

class SCEnv_Ld(SCEnv):
    def __init__(self, seed=None):
        super().__init__((2, 1), False, False, 1, 4, seed)

class SCEnv_LdH(SCEnv):
    def __init__(self, seed=None):
        super().__init__((2, 1), True, False, 1, 4, seed)

class SCEnv_LdHB(SCEnv):
    def __init__(self, seed=None):
        super().__init__((2, 1), True, True, 1, 4, seed)

class SCEnv_1Q(SCEnv):
    def __init__(self, seed=None):
        super().__init__((1, 1), True, True, 1, 5, seed)

class SCEnv_2Q(SCEnv):
    def __init__(self, seed=None):
        super().__init__((1, 1), True, True, 2, 11, seed)

class SCEnv_Full(SCEnv):
    def __init__(self, seed=None):
        super().__init__((1, 1), True, True, 4, 25, seed)

register(
    id="SC-Ld-v0",
    entry_point="envs:SCEnv_Ld"
)

register(
    id="SC-LdH-v0",
    entry_point="envs:SCEnv_LdH"
)

register(
    id="SC-LdHB-v0",
    entry_point="envs:SCEnv_LdHB"
)

register(
    id="SC-1Q-v0",
    entry_point="envs:SCEnv_1Q"
)

register(
    id="SC-2Q-v0",
    entry_point="envs:SCEnv_2Q"
)

register(
    id="SC-Full-v0",
    entry_point="envs:SCEnv_Full"
)