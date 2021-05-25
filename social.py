from magent.gridworld import Config, CircleRange, AgentSymbol, Event

DEER_AGENT_TYPE: str = "deer"
TIGER_AGENT_TYPE: str = "tiger"


def get_forest_configuration(map_size: int) -> Config:
    configuration: Config = Config()

    configuration.set({"map_width": map_size,
                       "map_height": map_size})
    configuration.set({"embedding_size": 10})

    deer_agent_type: str = configuration.register_agent_type(DEER_AGENT_TYPE, {'width': 1,
                                                                               'length': 1,
                                                                               'hp': 5,
                                                                               'speed': 1,
                                                                               'view_range': CircleRange(1),
                                                                               'attack_range': CircleRange(0),
                                                                               'damage': 0,
                                                                               'step_recover': 0.2,
                                                                               'food_supply': 0,
                                                                               'kill_supply': 8,
                                                                               # added to standard 'forest' setup to motivate deers to live longer :)
                                                                               'step_reward': 1})
    ethical_tiger_agent_type: str = configuration.register_agent_type(TIGER_AGENT_TYPE, {'width': 1,
                                                                                         'length': 1,
                                                                                         'hp': 10,
                                                                                         'speed': 1,
                                                                                         'view_range': CircleRange(4),
                                                                                         'attack_range': CircleRange(1),
                                                                                         'damage': 3,
                                                                                         'step_recover': -0.5,
                                                                                         'food_supply': 0,
                                                                                         'kill_supply': 0, # Temporarily disabled.
                                                                                         'step_reward': 1,
                                                                                         'attack_penalty': -0.1})

    configuration.add_group(deer_agent_type)
    configuration.add_group(ethical_tiger_agent_type)

    return configuration


def get_social_configuration(map_size: int) -> Config:
    social_configuration: Config = get_forest_configuration(map_size)

    tiger_group_handle: int = 1

    tiger_agent: AgentSymbol = AgentSymbol(tiger_group_handle, index='any')
    another_tiger_agent: AgentSymbol = AgentSymbol(tiger_group_handle, index='any')

    tiger_attacking: Event = Event(tiger_agent, 'attack', another_tiger_agent)
    social_configuration.add_reward_rule(tiger_attacking, receiver=[tiger_agent, another_tiger_agent],
                                         value=[-0.5, -0.5])

    tiger_defending: Event = Event(another_tiger_agent, 'attack', tiger_agent)
    social_configuration.add_reward_rule(tiger_attacking & tiger_defending, receiver=[tiger_agent, another_tiger_agent],
                                         value=[-0.5, +0.4])
    return social_configuration
