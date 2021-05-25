from magent.gridworld import Config, CircleRange


def get_forest_configuration(map_size: int) -> Config:
    configuration: Config = Config()

    configuration.set({"map_width": map_size,
                       "map_height": map_size})
    configuration.set({"embedding_size": 10})

    deer_agent_type: str = configuration.register_agent_type("deer", {'width': 1,
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
    ethical_tiger_agent_type: str = configuration.register_agent_type("ethical_tiger", {'width': 1,
                                                                                        'length': 1,
                                                                                        'hp': 10,
                                                                                        'speed': 1,
                                                                                        'view_range': CircleRange(4),
                                                                                        'attack_range': CircleRange(1),
                                                                                        'damage': 3,
                                                                                        'step_recover': -0.5,
                                                                                        'food_supply': 0,
                                                                                        'kill_supply': 16,
                                                                                        'step_reward': 1,
                                                                                        'attack_penalty': -0.1})

    configuration.add_group(deer_agent_type)
    configuration.add_group(ethical_tiger_agent_type)

    return configuration
