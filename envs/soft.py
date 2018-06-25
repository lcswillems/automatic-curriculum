from gym.envs.registration import register
from .general import BaseEnv

class SoftSCEnv(BaseEnv):
    def __init__(self, in_box, blocked, seed=None):
        self.in_box = in_box
        self.blocked = blocked

        super().__init__(
            num_cols=2,
            num_rows=1,
            num_rooms_visited=2,
            seed=None
        )

    def gen_mission(self):
        super().gen_mission()

        self.add_door(0, 0, locked=True, door_idx=0,
                      color=self.door_colors[0],
                      in_box=self.in_box,
                      blocked=self.blocked)

        self.add_object(1, 0, "ball", color=self.ball_to_find_color)
        self.place_agent(0, 0)

class SoftSCEnv_Ld(SoftSCEnv):
    def __init__(self, seed=None):
        super().__init__(False, False, seed)

class SoftSCEnv_LdH(SoftSCEnv):
    def __init__(self, seed=None):
        super().__init__(True, False, seed)

class SoftSCEnv_LdHB(SoftSCEnv):
    def __init__(self, seed=None):
        super().__init__(True, True, seed)

register(
    id="SC-Soft-Ld-v0",
    entry_point="envs:SoftSCEnv_Ld"
)

register(
    id="SC-Soft-LdH-v0",
    entry_point="envs:SoftSCEnv_LdH"
)

register(
    id="SC-Soft-LdHB-v0",
    entry_point="envs:SoftSCEnv_LdHB"
)