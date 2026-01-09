import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 2  # Bodyguards
        num_adversaries = 2  # Assassins
        num_agents = num_good_agents + num_adversaries + 1  # +1 for VIP
        num_landmarks = 1  # The target the VIP must reach

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.collide = True
            agent.silent = True

            # Agent 0 is the VIP
            if i == 0:
                agent.adversary = False
                agent.name = "VIP"
                agent.size = 0.075
                agent.accel = 3.0
                agent.max_speed = 0.8  # Slow!
            # Next 'num_good_agents' are Bodyguards
            elif i < 1 + num_good_agents:
                agent.adversary = False
                agent.name = "Bodyguard %d" % i
                agent.size = 0.05
                agent.accel = 3.0
                agent.max_speed = 1.3  # Faster than VIP
            # The rest are Adversaries
            else:
                agent.adversary = True
                agent.name = "Assassin %d" % i
                agent.size = 0.05
                agent.accel = 4.0
                agent.max_speed = 1.3  # Agile

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.1

        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.35, 0.35, 0.85])
                if not agent.adversary
                else np.array([0.85, 0.35, 0.35])
            )
            if agent.name == "VIP":
                agent.color = np.array([0.35, 0.85, 0.35])  # Green VIP

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        return self.reward(agent, world)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )

    def agent_reward(self, agent, world):
        # Good agents (VIP + Bodyguards) share a common goal
        vip = world.agents[0]
        target = world.landmarks[0]

        # 1. Distance from VIP to Target (We want this minimized)
        dist_to_target = np.sqrt(
            np.sum(np.square(vip.state.p_pos - target.state.p_pos))
        )
        rew = -0.1 * dist_to_target

        # 2. Penalty if VIP is hit by assassin
        collisions = 0
        for adv in world.agents:
            if adv.adversary and self.is_collision(vip, adv):
                collisions += 1

        # Heavy penalty for collision
        rew -= 10 * collisions

        # Small reward for bodyguards staying near VIP (to encourage formation)
        if agent != vip:
            dist_to_vip = np.sqrt(
                np.sum(np.square(agent.state.p_pos - vip.state.p_pos))
            )
            rew -= 0.05 * dist_to_vip

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries want to hit the VIP
        vip = world.agents[0]
        rew = 0

        if self.is_collision(agent, vip):
            rew += 10

        # Optional: shape reward to encourage chasing
        dist_to_vip = np.sqrt(
            np.sum(np.square(agent.state.p_pos - vip.state.p_pos))
        )
        rew -= 0.1 * dist_to_vip

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)

        # VIP needs to know where the target is!
        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
            + other_pos
            + other_vel
        )
