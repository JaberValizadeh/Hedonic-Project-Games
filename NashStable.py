import math
from itertools import product
from typing import Dict, List, Set, Tuple, Any
from HedonicProjectGame import HedonicProjectGame, create_example_1

class NashStable(HedonicProjectGame):
    """
    Implementation of Nash Stability for Project Hedonic Games.
    
    A strategy profile s is Nash stable (NS) if, for all players i ∈ N and all j ∈ M:
    u_i(s) ≥ u_i(j, s_{-i})
    
    where (j, s_{-i}) denotes the profile obtained after player i deviates from s_i to j.
    """
    
    def __init__(self, players: List[int], projects: List[Any], rewards: Dict[Any, float], 
                 preferences: Dict[int, Dict[frozenset, float]]):
        """
        Initialize a Nash Stable Project Hedonic Game.
        
        Args:
            players: List of player IDs (N)
            projects: List of project IDs (M) 
            rewards: Dictionary mapping project to reward value (r function)
            preferences: Dictionary mapping player to their preference function over coalitions
        """
        super().__init__(players, projects, rewards, preferences)
    
    def create_deviation_profile(self, strategy_profile: Dict[int, Any], 
                                deviating_player: int, new_project: Any) -> Dict[int, Any]:
        """
        Create a new strategy profile after a player deviation.
        
        Args:
            strategy_profile: Current strategy profile s
            deviating_player: Player i who is deviating
            new_project: Project j that player i moves to
            
        Returns:
            New strategy profile (j, s_{-i})
        """
        deviation_profile = strategy_profile.copy()
        deviation_profile[deviating_player] = new_project
        return deviation_profile
    
    def is_nash_stable(self, strategy_profile: Dict[int, Any]) -> bool:
        """
        Check if a strategy profile is Nash stable according to the definition:
        
        A strategy profile s is Nash stable if, for all players i ∈ N and all j ∈ M:
        u_i(s) ≥ u_i(j, s_{-i})
        
        Args:
            strategy_profile: The strategy profile to check
            
        Returns:
            True if the profile is Nash stable, False otherwise
        """
        for player in self.players:
            current_utility = self.compute_utility(strategy_profile, player)
            current_project = strategy_profile[player]
            
            # Check all possible deviations for this player
            for project in self.projects:
                if project == current_project:
                    continue  # Skip current project (no deviation)
                
                # Create deviation profile (j, s_{-i})
                deviation_profile = self.create_deviation_profile(
                    strategy_profile, player, project
                )
                
                # Compute utility after deviation
                deviation_utility = self.compute_utility(deviation_profile, player)
                
                # Check Nash stability condition: u_i(s) ≥ u_i(j, s_{-i})
                if deviation_utility > current_utility + 1e-10:
                    # Player has incentive to deviate - not Nash stable
                    return False
        
        return True
    
    def find_all_nash_stable(self) -> List[Dict[int, Any]]:
        """
        Find all Nash stable strategy profiles.
        
        Returns:
            List of all Nash stable profiles
        """
        nash_stable_profiles = []
        all_profiles = self.generate_all_strategy_profiles()
        
        for profile in all_profiles:
            if self.is_nash_stable(profile):
                nash_stable_profiles.append(profile)
        
        return nash_stable_profiles
    
    def analyze_deviation_incentives(self, strategy_profile: Dict[int, Any]) -> Dict[int, Dict[Any, float]]:
        """
        Analyze deviation incentives for each player in a given strategy profile.
        
        Args:
            strategy_profile: The strategy profile to analyze
            
        Returns:
            Dictionary mapping player to their deviation incentives
            Format: {player: {project: utility_gain}}
        """
        deviation_incentives = {}
        
        for player in self.players:
            current_utility = self.compute_utility(strategy_profile, player)
            current_project = strategy_profile[player]
            player_incentives = {}
            
            for project in self.projects:
                if project == current_project:
                    continue
                
                deviation_profile = self.create_deviation_profile(
                    strategy_profile, player, project
                )
                deviation_utility = self.compute_utility(deviation_profile, player)
                utility_gain = deviation_utility - current_utility
                
                player_incentives[project] = utility_gain
            
            deviation_incentives[player] = player_incentives
        
        return deviation_incentives
    
    def print_nash_analysis(self, strategy_profile: Dict[int, Any]):
        """
        Print detailed Nash stability analysis for a strategy profile.
        
        Args:
            strategy_profile: The strategy profile to analyze
        """
        print(f"\nNash Stability Analysis for profile {strategy_profile}")
        print("=" * 60)
        
        is_nash = self.is_nash_stable(strategy_profile)
        print(f"Nash Stable: {'YES' if is_nash else 'NO'}")
        
        # Show current utilities
        current_utilities = self.compute_all_utilities(strategy_profile)
        print(f"\nCurrent utilities:")
        for player, utility in current_utilities.items():
            current_project = strategy_profile[player]
            print(f"  Player {player} (project {current_project}): {utility:.4f}")
        
        # Show deviation incentives
        incentives = self.analyze_deviation_incentives(strategy_profile)
        print(f"\nDeviation incentives:")
        
        for player in self.players:
            current_project = strategy_profile[player]
            print(f"  Player {player} (currently in {current_project}):")
            
            player_incentives = incentives[player]
            has_positive_incentive = any(gain > 1e-10 for gain in player_incentives.values())
            
            if not has_positive_incentive:
                print(f"    No profitable deviations")
            else:
                for project, gain in player_incentives.items():
                    if gain > 1e-10:
                        deviation_profile = self.create_deviation_profile(
                            strategy_profile, player, project
                        )
                        new_utility = self.compute_utility(deviation_profile, player)
                        print(f"    → {project}: gain = +{gain:.4f} (new utility: {new_utility:.4f})")
    
    def compare_stability_concepts(self, strategy_profile: Dict[int, Any]) -> Dict[str, bool]:
        """
        Compare different stability concepts for a given profile.
        Note: This is a placeholder for comparison with other stability concepts.
        
        Args:
            strategy_profile: The strategy profile to analyze
            
        Returns:
            Dictionary showing which stability concepts are satisfied
        """
        result = {
            'Nash Stable': self.is_nash_stable(strategy_profile)
        }
        
        # Placeholder for other stability concepts
        # These would need to be implemented in their respective classes
        result['Protective Stable'] = None  # Would need PS implementation
        result['Group Stable'] = None       # Would need GS implementation
        
        return result


def create_nash_example_1():
    """
    Create Example 1 as a Nash Stable game.
    """
    players = [1, 2, 3, 4]
    projects = ['a', 'b', 'c', 'd']
    rewards = {'a': 10, 'b': 5, 'c': 5, 'd': 5}
    
    preferences = {
        1: {
            frozenset([1]): 0.0,
            frozenset([1, 2]): 0.167,
            frozenset([1, 3]): 0.042,
            frozenset([1, 4]): 0.042,
            frozenset([1, 2, 3]): 0.208,
            frozenset([1, 2, 4]): 0.208,
            frozenset([1, 3, 4]): 0.083,
            frozenset([1, 2, 3, 4]): 0.25
        },
        2: {
            frozenset([2]): 0.0,
            frozenset([1, 2]): 0.143,
            frozenset([2, 3]): 0.1,
            frozenset([2, 4]): 0.167,
            frozenset([1, 2, 3]): 0.214,
            frozenset([1, 2, 4]): 0.178,
            frozenset([2, 3, 4]): 0.107,
            frozenset([1, 2, 3, 4]): 0.25
        },
        3: {
            frozenset([3]): 0.0,
            frozenset([1, 3]): 0.0625,
            frozenset([2, 3]): 0.0125,
            frozenset([3, 4]): 0.0625,
            frozenset([1, 2, 3]): 0.187,
            frozenset([1, 3, 4]): 0.0125,
            frozenset([2, 3, 4]): 0.0187,
            frozenset([1, 2, 3, 4]): 0.25
        },
        4: {
            frozenset([4]): 0.0,
            frozenset([1, 4]): 0.0833,
            frozenset([2, 4]): 0.0833,
            frozenset([3, 4]): 0.0833,
            frozenset([1, 2, 4]): 0.167,
            frozenset([1, 3, 4]): 0.167,
            frozenset([2, 3, 4]): 0.167,
            frozenset([1, 2, 3, 4]): 0.25
        }
    }
    
    return NashStable(players, projects, rewards, preferences)


def main():
    """Demonstrate Nash stability analysis."""
    # Create the game
    game = create_nash_example_1()
    
    # Analyze all strategy profiles
    all_profiles = game.generate_all_strategy_profiles()
    print(f"Analyzing all {len(all_profiles)} possible strategy profiles:")
    
    print(f"\nProfile      | Nash  | u_1    | u_2    | u_3    | u_4    | Total")
    print(f"-------------|-------|--------|--------|--------|--------|--------")
    
    nash_count = 0
    for profile in all_profiles:
        is_nash = game.is_nash_stable(profile)
        if is_nash:
            nash_count += 1
        
        utilities = game.compute_all_utilities(profile)
        total_utility = sum(utilities.values())
        profile_str = f"({profile[1]}, {profile[2]}, {profile[3]}, {profile[4]})"
        nash_str = "YES" if is_nash else "NO"
        
        print(f"{profile_str:12} | {nash_str:5} | {utilities[1]:6.4f} | {utilities[2]:6.4f} | {utilities[3]:6.4f} | {utilities[4]:6.4f} | {total_utility:6.4f}")
    
    print(f"\nSummary: {nash_count}/{len(all_profiles)} profiles are Nash stable")
    
    # List all Nash stable outcomes
    nash_stable_profiles = game.find_all_nash_stable()
    print(f"\nNash Stable Outcomes:")
    print("=" * 40)
    if nash_stable_profiles:
        for i, profile in enumerate(nash_stable_profiles, 1):
            profile_str = f"({profile[1]}, {profile[2]}, {profile[3]}, {profile[4]})"
            utilities = game.compute_all_utilities(profile)
            total_utility = sum(utilities.values())
            print(f"{i}. {profile_str} - Total utility: {total_utility:.4f}")
            print(f"   Individual utilities: u_1={utilities[1]:.4f}, u_2={utilities[2]:.4f}, u_3={utilities[3]:.4f}, u_4={utilities[4]:.4f}")
    else:
        print("No Nash stable outcomes found.")


if __name__ == "__main__":
    main()
