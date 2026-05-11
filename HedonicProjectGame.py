import math
from itertools import product
from typing import Dict, List, Set, Tuple, Any

class HedonicProjectGame:
    
    def __init__(self, players: List[int], projects: List[Any], rewards: Dict[Any, float], 
                 preferences: Dict[int, Dict[frozenset, float]]):

        self.players = players
        self.projects = projects
        self.rewards = rewards
        self.preferences = preferences
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate that the game parameters are correctly specified."""
        # Check that all projects have rewards
        for project in self.projects:
            if project not in self.rewards:
                raise ValueError(f"Project {project} missing reward")
            if self.rewards[project] <= 0:
                raise ValueError(f"Project {project} reward must be positive")
        
        # Check that preference functions are normalized
        for player in self.players:
            if player not in self.preferences:
                raise ValueError(f"Player {player} missing preferences")
            
            # Check normalization: sum of preferences should be 1
            total_pref = sum(self.preferences[player].values())
            if abs(total_pref - 1.0) > 1e-10:
                print(f"Warning: Player {player} preferences not normalized (sum = {total_pref:.6f})")
            
            # Check that all coalitions containing the player are specified
            player_coalitions = self._get_possible_coalitions_for_player(player)
            for coalition in player_coalitions:
                if coalition not in self.preferences[player]:
                    raise ValueError(f"Player {player} missing preference for coalition {set(coalition)}")
    
    def _get_possible_coalitions_for_player(self, player: int) -> List[frozenset]:
        """Get all possible coalitions that contain the given player (N_i)."""
        other_players = [p for p in self.players if p != player]
        coalitions = []
        
        # Generate all subsets of other players and add the player to each
        for size in range(len(other_players) + 1):
            for subset in self._combinations(other_players, size):
                coalition = frozenset([player] + list(subset))
                coalitions.append(coalition)
        
        return coalitions
    
    def _combinations(self, items: List, r: int) -> List[Tuple]:
        """Generate combinations of items taken r at a time."""
        if r == 0:
            return [()]
        if not items:
            return []
        
        first = items[0]
        rest = items[1:]
        
        # Combinations that include the first item
        with_first = [(first,) + combo for combo in self._combinations(rest, r-1)]
        # Combinations that don't include the first item  
        without_first = self._combinations(rest, r)
        
        return with_first + without_first
    
    def get_coalition_for_project(self, strategy_profile: Dict[int, Any], project: Any) -> Set[int]:
        """
        Get the coalition C_j(s) for a given project j under strategy profile s.
        
        Args:
            strategy_profile: Dictionary mapping player to chosen project
            project: The project to get coalition for
            
        Returns:
            Set of players who chose the given project
        """
        return {player for player, chosen_project in strategy_profile.items() 
                if chosen_project == project}
    
    def compute_utility(self, strategy_profile: Dict[int, Any], player: int) -> float:
        """
        Compute utility for a player under a given strategy profile using Equation (1):
        u_i(s) = p_i(C_{s_i}(s)) * r(s_i) / |C_{s_i}(s)|
        
        Args:
            strategy_profile: Dictionary mapping player to chosen project
            player: The player to compute utility for
            
        Returns:
            Utility value for the player
        """
        if player not in strategy_profile:
            raise ValueError(f"Player {player} not in strategy profile")
        
        chosen_project = strategy_profile[player]
        coalition = self.get_coalition_for_project(strategy_profile, chosen_project)
        coalition_size = len(coalition)
        
        # Get player's preference for this coalition
        coalition_frozen = frozenset(coalition)
        if coalition_frozen not in self.preferences[player]:
            raise ValueError(f"Player {player} has no preference for coalition {set(coalition)}")
        
        preference_value = self.preferences[player][coalition_frozen]
        project_reward = self.rewards[chosen_project]
        
        # Apply Equation (1)
        utility = preference_value * (project_reward / coalition_size)
        
        return utility
    
    def compute_all_utilities(self, strategy_profile: Dict[int, Any]) -> Dict[int, float]:
        """Compute utilities for all players under a given strategy profile."""
        utilities = {}
        for player in self.players:
            utilities[player] = self.compute_utility(strategy_profile, player)
        return utilities
    
    def generate_all_strategy_profiles(self) -> List[Dict[int, Any]]:
        """Generate all possible strategy profiles."""
        profiles = []
        for assignment in product(self.projects, repeat=len(self.players)):
            profile = {player: project for player, project in zip(self.players, assignment)}
            profiles.append(profile)
        return profiles
    
    def is_nash_stable(self, strategy_profile: Dict[int, Any]) -> bool:
        """Check if a strategy profile is Nash Stable (no player can improve by deviating)."""
        for player in self.players:
            current_utility = self.compute_utility(strategy_profile, player)
            current_project = strategy_profile[player]
            
            # Check all alternative projects
            for alt_project in self.projects:
                if alt_project == current_project:
                    continue
                
                # Create deviation profile
                deviation = strategy_profile.copy()
                deviation[player] = alt_project
                deviation_utility = self.compute_utility(deviation, player)
                
                if deviation_utility > current_utility:
                    return False
        return True
    
    def is_lonely_stable(self, strategy_profile: Dict[int, Any]) -> bool:
        """Check if profile is Lonely Stable (no player prefers to work alone on any project)."""
        for player in self.players:
            current_utility = self.compute_utility(strategy_profile, player)
            
            # Check if player prefers being alone on any project
            for project in self.projects:
                # Simulate player being alone on this project
                alone_coalition = frozenset([player])
                alone_utility = self.preferences[player][alone_coalition] * self.rewards[project]
                
                if alone_utility > current_utility:
                    return False
        return True
    
    def is_jump_stable(self, strategy_profile: Dict[int, Any]) -> bool:
        """Check if profile is Jump Stable (no player wants to jump to join another non-empty coalition)."""
        for player in self.players:
            current_utility = self.compute_utility(strategy_profile, player)
            current_project = strategy_profile[player]
            
            # Check jumping to other projects that have existing coalitions
            for alt_project in self.projects:
                if alt_project == current_project:
                    continue
                
                existing_coalition = self.get_coalition_for_project(strategy_profile, alt_project)
                if not existing_coalition:  # Skip empty projects (that's lonely deviation)
                    continue
                
                # Player joins existing coalition
                deviation = strategy_profile.copy()
                deviation[player] = alt_project
                deviation_utility = self.compute_utility(deviation, player)
                
                if deviation_utility > current_utility:
                    return False
        return True
    
    def print_game_info(self):
        """Print basic information about the game."""
        print(f"Project Hedonic Game:")
        print(f"  Players: {self.players}")
        print(f"  Projects: {self.projects}")
        print(f"  Rewards: {self.rewards}")
        print(f"  Strategy space size: {len(self.projects)}^{len(self.players)} = {len(self.projects)**len(self.players)}")
    
    def print_strategy_profile_analysis(self, strategy_profile: Dict[int, Any]):
        """Print detailed analysis of a strategy profile."""
        print(f"\nStrategy Profile: {strategy_profile}")
        
        # Show coalitions for each project
        print("Coalitions by project:")
        for project in self.projects:
            coalition = self.get_coalition_for_project(strategy_profile, project)
            if coalition:
                print(f"  Project {project}: {set(coalition)} (size: {len(coalition)})")
            else:
                print(f"  Project {project}: [] (empty)")
        
        # Show utilities
        utilities = self.compute_all_utilities(strategy_profile)
        print("Player utilities:")
        for player in self.players:
            chosen_project = strategy_profile[player]
            coalition = self.get_coalition_for_project(strategy_profile, chosen_project)
            coalition_size = len(coalition)
            preference = self.preferences[player][frozenset(coalition)]
            reward = self.rewards[chosen_project]
            
            print(f"  Player {player}: u_{player}(s) = {preference:.2f} × {reward}/{coalition_size} = {utilities[player]:.4f}")


def create_example_1():

    players = [1, 2]
    projects = ['a', 'b']
    rewards = {'a': 2, 'b': 2}
    
    # Define preference functions for each player
    preferences = {
        1: {
            frozenset([1]): 0.001,
            frozenset([1, 2]): 0.999,
        },
        2: {
            frozenset([2]): 0.001,
            frozenset([1, 2]): 0.999,
        }
    }
    
    return HedonicProjectGame(players, projects, rewards, preferences)


def main():
    """Demonstrate the Project Hedonic Game with Example 1."""
    # Create Example 1 from the paper
    game = create_example_1()
    
    # Analyze all possible strategy profiles
    all_profiles = game.generate_all_strategy_profiles()
    
    print(f"Utility analysis for all profiles:")
    print(f"Profile      | u_1(.) | u_2(.) | SW(.)  | NS? | LS? | JS?")
    print(f"-------------|--------|--------|--------|-----|-----|-----")
    
    for profile in all_profiles:
        utilities = game.compute_all_utilities(profile)
        total_utility = sum(utilities.values())
        is_ns = game.is_nash_stable(profile)
        is_ls = game.is_lonely_stable(profile)
        is_js = game.is_jump_stable(profile)
        profile_str = f"({profile[1]}, {profile[2]})"
        print(f"{profile_str:12} | {utilities[1]:6.4f} | {utilities[2]:6.4f} | {total_utility:6.4f} | {'Y' if is_ns else 'N':3} | {'Y' if is_ls else 'N':3} | {'Y' if is_js else 'N':3}")


def demo_basic():
    """Simple demonstration function for testing."""
    print("Basic Project Hedonic Game Demo")
    print("================================")
    
    game = create_example_1()
    print(f"Created game with {len(game.players)} players and {len(game.projects)} projects")
    
    # Test initial profile
    initial_profile = {1: 'a', 2: 'a'}
    utilities = game.compute_all_utilities(initial_profile)
    
    print(f"\nStrategy profile (a,a):")
    print(f"Coalition for project a: {game.get_coalition_for_project(initial_profile, 'a')}")
    print(f"Coalition for project b: {game.get_coalition_for_project(initial_profile, 'b')}")
    
    print(f"\nUtilities:")
    for player, utility in utilities.items():
        print(f"  Player {player}: {utility:.4f}")
    
    # Expected: u_i = p_i({1,2}) * r(a) / 2 = 0.999 * 99 / 2 = 49.4505
    print(f"\nVerification:")
    expected = {1: 0.999 * 99 / 2, 2: 0.999 * 99 / 2}
    for player in [1, 2]:
        exp = expected[player]
        comp = utilities[player]
        match = "✓" if abs(exp - comp) < 1e-6 else "✗"
        print(f"  Player {player}: Expected {exp:.4f}, Computed {comp:.4f} {match}")


if __name__ == "__main__":
    main()
