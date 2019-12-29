class Playback():
    class Episode():
        def __init__(self):
            self.actors_k = []
            self.states_k_sum = []
            self.states_k_showing_card_value = []
            self.states_k_has_usable_ace = []
            self.actions_k = []
            self.rewards_k_plus_1 = []
            
        def preprocess(self):
            # re-assign final reward to last player turn
            
            # remove dealer turns
            pass
            
    def __init__(self):
        """Episodes for t=0, t=1, ..., t=T-1"""
        self.episodes = []
        
    def start_episode(self):
        self.episodes.append(Playback.Episode())
        
    def register_actor(self, is_player: bool):
        self.episodes[-1].actors_k.append(is_player)
        
    def register_state(self, player_sum: int, dealer_showing_card_value: int, player_has_usable_ace: bool):
        self.episodes[-1].states_k_sum.append(player_sum)
        self.episodes[-1].states_k_showing_card_value.append(dealer_showing_card_value)
        self.episodes[-1].states_k_has_usable_ace.append(player_has_usable_ace)
        
    def register_action(self, action_type: int):
        self.episodes[-1].actions_k.append(action_type)
        
    def register_reward(self, reward_value: int):
        self.episodes[-1].rewards_k_plus_1.append(reward_value)
        
    def save(self):
        pass
    
    def load(self):
        pass
        
playback = Playback()