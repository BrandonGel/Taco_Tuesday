"""
    using a-star algorithm to find trajectory and capture flag

    ** The method only works under full observation setting
"""

import numpy as np
import gym_cap.envs.const as const

from .policy import Policy


class Invariant(Policy):
    """Policy generator class for CtF env.
    
    This class can be used as a template for policy generator.
    Designed to summon an AI logic for the team of units.

    Policy: Get the Flag
    
    Methods:
        gen_action: Required method to generate a list of actions.
    """

    def __init__(self):
        """Constuctor for policy class.
        
        This class can be used as a template for policy generator.
        
        Args:
            free_map (np.array): 2d map of static environment.
            agent_list (list): list of all friendly units.
        """

        super().__init__()

        # Initial Parameters
        self.agent_type = None
        self.team = None
        self.free_map = None
        self.random = None

        # Defense Parameters
        self.flag_location = None
        self.flag_code = None
        self.exploration = None

        # Patrol Variable
        self.route = None
        self.assigned = None
        self.grouped_boarder = None
        self.heading_right = None

        # Astar Variable
        self._random_transition_safe = False
        self.agent_route = None
        self.found_route = None
        self.agent_steps = None

    def initiate(self, free_map, agent_list):
        super().initiate(free_map, agent_list)
        self.free_map = free_map
        self.team = agent_list[0].team
        self.random = np.random
        self.agent_type = {}
        self.primitives = len(agent_list)*[[]]  #History of each agent primitives

        # names = np.random.choice(['astar', 'patrol', 'def'], len(agent_list))
        # names = 30 * ['astar']
        # names = 30 * ['patrol']
        names = 30 * ['def']
        # names = 10*['astar', 'patrol', 'def']
        for i in range(len(agent_list)):
            self.agent_type.update({agent_list[i]: names[i]})

        ##Defense
        #12/31/22: Update defense to partial blue observation
        self.flag_location = self.get_flag_loc(self.team, True)
        # self.flag_location = None
        self.exploration = 0.1
        self.flag_code = const.TEAM1_FLAG if agent_list[0].team == const.TEAM1_BACKGROUND else const.TEAM2_FLAG

        ##Patrol
        self.patrolInit(agent_list)

        ## Astar
        self.found_route = []
        self.agent_route = []
        flag_id = const.TEAM2_FLAG if agent_list[0].team == const.TEAM1_BACKGROUND else const.TEAM1_FLAG
        flag = tuple(np.argwhere(free_map == flag_id)[0])
        self.agent_steps = [0] * len(agent_list)
        for idx, agent in enumerate(agent_list):
            start = agent.get_loc()
            self.agent_route.append(self.route_astar(start, flag))
            self.found_route.append(self.agent_route[idx] is not None)

    def patrolInit(self, agent_list):
        other_team = const.TEAM2_BACKGROUND if self.team == const.TEAM1_BACKGROUND else const.TEAM1_BACKGROUND

        # Scan and list boarder location
        boarder = []
        for i in range(len(self.free_map)):
            for j in range(len(self.free_map[0])):
                if self.free_map[i][j] == self.team:
                    count = 0
                    for move in range(1, 5):
                        nx, ny = self.next_loc((i, j), move)
                        if nx < 0 or nx >= len(self.free_map): continue
                        if ny < 0 or ny >= len(self.free_map[0]): continue
                        if self.free_map[nx][ny] == other_team:
                            count += 1
                            break
                    if count:
                        boarder.append((i, j))

        # Group boarder (BFS)
        grouped_boarder = []
        while len(boarder) > 0:
            visited = []
            queue = []
            queue.append(boarder.pop())
            while len(queue) > 0:
                n = queue.pop()
                visited.append(tuple(n))
                for move in range(1, 5):
                    nx, ny = self.next_loc(n, move)
                    if (nx, ny) in boarder:
                        boarder.remove((nx, ny))
                        queue.append((nx, ny))
            grouped_boarder.append(visited)

        boarder_centroid = [np.mean(boarder, axis=0) for boarder in grouped_boarder]
        # Assign boarder
        self.assigned = []
        for agent in agent_list:
            x = np.asarray(agent.get_loc())
            dist = [sum(abs(x - centroid)) for centroid in boarder_centroid]  # L1 norm
            b = np.argmin(dist)
            self.assigned.append(b)

        # Find path to boarder
        self.route = []
        for idx, agent in enumerate(agent_list):
            target_idx = self.random.choice(len(grouped_boarder[self.assigned[idx]]))
            target = grouped_boarder[self.assigned[idx]][target_idx]
            route = self.route_astar(agent.get_loc(), target)
            if route is None:
                self.route.append(None)
            else:
                self.route.append(route)

        self.grouped_boarder = grouped_boarder
        self.heading_right = [True] * len(agent_list)  #: Attr to track directions.

    # def getFlag(self):
    #     flag_id = const.TEAM2_FLAG if agent_list[0].team == const.TEAM1_BACKGROUND else const.TEAM1_FLAG
    #     flag = tuple(np.argwhere(free_map == flag_id)[0])
    #     self.agent_steps = [0] * len(agent_list)
    #     for idx, agent in enumerate(agent_list):
    #         start = agent.get_loc()
    #         self.agent_route.append(self.route_astar(start, flag))
    #         self.found_route.append(self.agent_route[idx] is not None)

    def gen_action(self, agent_list, observation):
        """Action generation method.
        
        This is a required method that generates list of actions corresponding 
        to the list of units. 
        
        Args:
            agent_list (list): list of all friendly units.
            observation (np.array): 2d map of partially observable map.
            
        Returns:
            action_out (list): list of integers as actions selected for team.
        """
        print(agent_list[0].team)
        action_out = []
        # print("INVARIANT")
        for idx, agent in enumerate(agent_list):
            # Not Alive or No Route Found
            if not agent.isAlive or not self.found_route[idx]:
                action_out.append(0)
                self.primitives[idx].append("Dead")
                continue

            # Defense Primitive
            if self.agent_type[agent] == 'def':
                # if map changes then reset the flag location
                # search for a flag until finds it
                if self.flag_location is None:
                    self.flag_location = self.get_flag_loc_partobs( True)  # In case it is partial observation

                if self.flag_location is None:  # Random Search
                    action_out.append(self.random.randint(0, 5))
                else:
                    # go to the flag to defend it
                    a = self.flag_approach(agent,observation)
                    action_out.append(a)

            # Patrol Primitive
            elif self.agent_type[agent] == 'patrol':
                boarder = self.grouped_boarder[self.assigned[idx]]
                route = self.route[idx]
                cur_loc = agent.get_loc()
                if cur_loc in boarder:  ## Patrol
                    self.route[idx] = None
                    a = self.patrol(cur_loc, boarder, self.free_map)
                    action_out.append(a)
                elif route is None:
                    action_out.append(np.random.randint(5))
                else:  ## Navigate to boarder
                    step = route.index(cur_loc)
                    new_loc = route[step + 1]
                    action = self.move_toward2(cur_loc, new_loc,observation)
                    action_out.append(action)

            # Astar Primitive
            elif self.agent_type[agent] == 'astar':
                cur_loc = agent.get_loc()
                if self.agent_route[idx][self.agent_steps[idx]] != cur_loc:
                    self.agent_steps[idx] += 1
                cur_step = self.agent_steps[idx]
                if cur_step >= len(self.agent_route[idx]) - 1:  # 11/29/2022: Subtract 1 to avoid list out of bounds
                    action_out.append(0)
                    continue
                new_loc = self.agent_route[idx][cur_step + 1]

                if new_loc[1] - cur_loc[1] > 0:  # move right
                    action = 3
                elif new_loc[1] - cur_loc[1] < 0:  # move left
                    action = 1
                elif new_loc[0] - cur_loc[0] > 0:  # move down
                    action = 2
                elif new_loc[0] - cur_loc[0] < 0:  # move up
                    action = 4
                action_out.append(action)

            elif self.agent_type[agent] == 'atomic':
                action = 3 #Move Right
                action_out.append(action)

        return action_out

    def flag_approach(self, agent,obs): #Add obs (observation)
        """Generate 1 action for given agent object."""
        action = self.move_toward2(agent.get_loc(), self.flag_location,obs)
        if self.random.random() < self.exploration:
            action = self.random.randint(0, 5)

        return action

    def patrol(self, loc, boarder, obs):
        x, y = loc

        # patrol along the boarder.
        action = [0]
        for a in range(1, 5):
            nx, ny = self.next_loc(loc, a)
            if not self.can_move(loc, a): continue
            if (nx, ny) in boarder:
                action.append(a)
        return np.random.choice(action)
