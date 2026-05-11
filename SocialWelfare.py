import math
from itertools import product
from typing import Dict, List, Set, Tuple, Any, Optional
from HedonicProjectGame import HedonicProjectGame, create_example_1

class SocialWelfare(HedonicProjectGame):
    """
    Implementation of Social Welfare computation and Social Optimum finding for Project Hedonic Games.
    
    Social Welfare: SW(s) = Σ_{i ∈ N} u_i(s)
    Social Optimum: s* = argmax_{s ∈ S} SW(s)
    Optimal Value: OPT(G) = SW(s*)
    
    This class provides methods to:
    1. Compute social welfare for any strategy profile
    2. Find all social optimal strategy profiles
    3. Analyze welfare properties and distribution
    """
    
    def __init__(self, players: List[int], projects: List[Any], rewards: Dict[Any, float], 
                 preferences: Dict[int, Dict[frozenset, float]]):
        """
        Initialize a Social Welfare Project Hedonic Game.
        
        Args:
            players: List of player IDs (N)
            projects: List of project IDs (M) 
            rewards: Dictionary mapping project to reward value (r function)
            preferences: Dictionary mapping player to their preference function over coalitions
        """
        super().__init__(players, projects, rewards, preferences)
    
    def compute_social_welfare(self, strategy_profile: Dict[int, Any]) -> float:
        """
        Compute social welfare for a given strategy profile.
        
        Social welfare is defined as: SW(s) = Σ_{i ∈ N} u_i(s)
        
        Args:
            strategy_profile: Dictionary mapping player to chosen project
            
        Returns:
            Social welfare value (sum of all player utilities)
        """
        total_welfare = 0.0
        
        for player in self.players:
            utility = self.compute_utility(strategy_profile, player)
            total_welfare += utility
        
        return total_welfare
    
    def find_social_optimum(self) -> Tuple[List[Dict[int, Any]], float]:
        """
        Find all social optimal strategy profiles and the optimal welfare value.
        
        A social optimum is: s* = argmax_{s ∈ S} SW(s)
        The optimal value is: OPT(G) = SW(s*)
        
        Returns:
            Tuple of (optimal_profiles, optimal_value) where:
            - optimal_profiles: List of all strategy profiles that achieve maximum social welfare
            - optimal_value: The maximum social welfare value OPT(G)
        """
        all_profiles = self.generate_all_strategy_profiles()
        
        max_welfare = float('-inf')
        optimal_profiles = []
        
        # Compute welfare for each strategy profile
        for profile in all_profiles:
            welfare = self.compute_social_welfare(profile)
            
            if welfare > max_welfare:
                # Found new maximum
                max_welfare = welfare
                optimal_profiles = [profile]
            elif abs(welfare - max_welfare) < 1e-10:
                # Found another profile with same maximum welfare
                optimal_profiles.append(profile)
        
        return optimal_profiles, max_welfare
    
    def analyze_all_profiles_welfare(self) -> List[Tuple[Dict[int, Any], float, List[float]]]:
        """
        Analyze social welfare for all possible strategy profiles.
        
        Returns:
            List of tuples (profile, welfare, individual_utilities) sorted by welfare (descending)
        """
        all_profiles = self.generate_all_strategy_profiles()
        welfare_analysis = []
        
        for profile in all_profiles:
            welfare = self.compute_social_welfare(profile)
            utilities = [self.compute_utility(profile, player) for player in self.players]
            welfare_analysis.append((profile, welfare, utilities))
        
        # Sort by welfare (descending)
        welfare_analysis.sort(key=lambda x: x[1], reverse=True)
        
        return welfare_analysis
    
    def compute_welfare_distribution(self) -> Dict[str, Any]:
        """
        Compute welfare distribution statistics across all strategy profiles.
        
        Returns:
            Dictionary with welfare statistics
        """
        welfare_analysis = self.analyze_all_profiles_welfare()
        welfare_values = [welfare for _, welfare, _ in welfare_analysis]
        
        stats = {
            'max_welfare': max(welfare_values),
            'min_welfare': min(welfare_values),
            'mean_welfare': sum(welfare_values) / len(welfare_values),
            'welfare_range': max(welfare_values) - min(welfare_values),
            'num_profiles': len(welfare_values),
            'welfare_values': welfare_values
        }
        
        # Count profiles by welfare level
        welfare_counts = {}
        for welfare in welfare_values:
            welfare_rounded = round(welfare, 6)  # Round to avoid floating point issues
            welfare_counts[welfare_rounded] = welfare_counts.get(welfare_rounded, 0) + 1
        
        stats['welfare_distribution'] = welfare_counts
        
        return stats
    
    def get_stable_profiles_welfare(self) -> Dict[str, Any]:
        """
        Get welfare values for different stability concepts.
        
        Returns:
            Dictionary with stable profiles and their welfare values
        """
        stable_welfare = {}
        
        # Try to get Nash stable profiles
        try:
            from NashStable import NashStable
            nash_game = NashStable(self.players, self.projects, self.rewards, self.preferences)
            
            nash_profiles = []
            nash_welfare_values = []
            
            for profile in self.generate_all_strategy_profiles():
                if nash_game.is_nash_stable(profile):
                    nash_profiles.append(profile)
                    nash_welfare_values.append(self.compute_social_welfare(profile))
            
            stable_welfare['nash'] = {
                'profiles': nash_profiles,
                'welfare_values': nash_welfare_values,
                'count': len(nash_profiles),
                'max_welfare': max(nash_welfare_values) if nash_welfare_values else 0,
                'min_welfare': min(nash_welfare_values) if nash_welfare_values else 0
            }
            
        except ImportError:
            stable_welfare['nash'] = None
        
        # Try to get Group stable profiles
        try:
            from GroupStable import GroupStable
            group_game = GroupStable(self.players, self.projects, self.rewards, self.preferences)
            
            group_profiles = group_game.find_all_group_stable()
            group_welfare_values = [self.compute_social_welfare(profile) for profile in group_profiles]
            
            stable_welfare['group'] = {
                'profiles': group_profiles,
                'welfare_values': group_welfare_values,
                'count': len(group_profiles),
                'max_welfare': max(group_welfare_values) if group_welfare_values else 0,
                'min_welfare': min(group_welfare_values) if group_welfare_values else 0
            }
            
        except ImportError:
            stable_welfare['group'] = None
        
        # Try to get Protective stable profiles
        try:
            from ProtectiveStable import ProtectiveStable
            protective_game = ProtectiveStable(self.players, self.projects, self.rewards, self.preferences)
            
            protective_profiles = protective_game.find_all_protective_stable()
            protective_welfare_values = [self.compute_social_welfare(profile) for profile in protective_profiles]
            
            stable_welfare['protective'] = {
                'profiles': protective_profiles,
                'welfare_values': protective_welfare_values,
                'count': len(protective_profiles),
                'max_welfare': max(protective_welfare_values) if protective_welfare_values else 0,
                'min_welfare': min(protective_welfare_values) if protective_welfare_values else 0
            }
            
        except ImportError:
            stable_welfare['protective'] = None
        
        return stable_welfare
    
    def print_social_welfare_analysis(self):
        """Print comprehensive social welfare analysis."""
        print("=" * 70)
        print("SOCIAL WELFARE ANALYSIS")
        print("=" * 70)
        
        # Find social optimum
        optimal_profiles, optimal_welfare = self.find_social_optimum()
        
        print(f"Social Optimum Analysis:")
        print(f"  Optimal Social Welfare: OPT(G) = {optimal_welfare:.4f}")
        print(f"  Number of optimal profiles: {len(optimal_profiles)}")
        
        print(f"\nOptimal Strategy Profiles:")
        for i, profile in enumerate(optimal_profiles, 1):
            print(f"  Profile {i}: {profile}")
            
            # Show coalitions and individual utilities
            print(f"    Coalitions:")
            total_check = 0
            for project in self.projects:
                coalition = self.get_coalition_for_project(profile, project)
                if coalition:
                    print(f"      Project {project}: {set(coalition)} (size: {len(coalition)})")
                    for player in coalition:
                        utility = self.compute_utility(profile, player)
                        total_check += utility
                        print(f"        Player {player}: u_{player} = {utility:.4f}")
                else:
                    print(f"      Project {project}: [] (empty)")
            print(f"    Total welfare verification: {total_check:.4f}")
        
        # Welfare distribution analysis
        print(f"\n" + "="*70)
        print("WELFARE DISTRIBUTION ANALYSIS")
        print("="*70)
        
        welfare_stats = self.compute_welfare_distribution()
        print(f"Welfare Statistics across all {welfare_stats['num_profiles']} strategy profiles:")
        print(f"  Maximum welfare: {welfare_stats['max_welfare']:.4f}")
        print(f"  Minimum welfare: {welfare_stats['min_welfare']:.4f}")
        print(f"  Average welfare: {welfare_stats['mean_welfare']:.4f}")
        print(f"  Welfare range: {welfare_stats['welfare_range']:.4f}")
        
        # Show top profiles
        welfare_analysis = self.analyze_all_profiles_welfare()
        print(f"\nTop 5 Strategy Profiles by Social Welfare:")
        print(f"{'Rank':<4} | {'Profile':<12} | {'Welfare':<8} | {'Individual Utilities'}")
        print("-" * 65)
        
        for rank, (profile, welfare, utilities) in enumerate(welfare_analysis[:5], 1):
            profile_str = f"({', '.join(str(profile[p]) for p in self.players)})"
            utilities_str = f"({', '.join(f'{u:.3f}' for u in utilities)})"
            print(f"{rank:<4} | {profile_str:<12} | {welfare:<8.4f} | {utilities_str}")
        
        # Show stable profiles welfare
        print(f"\n" + "="*70)
        print("STABLE PROFILES WELFARE")
        print("="*70)
        
        stable_welfare = self.get_stable_profiles_welfare()
        
        print(f"Social Optimum vs Stable Profiles:")
        print(f"  Optimal welfare: {optimal_welfare:.4f}")
        
        if stable_welfare.get('nash'):
            nash_data = stable_welfare['nash']
            print(f"  Nash stable profiles: {nash_data['count']}")
            if nash_data['welfare_values']:
                print(f"    Best Nash welfare: {nash_data['max_welfare']:.4f}")
                print(f"    Worst Nash welfare: {nash_data['min_welfare']:.4f}")
        
        if stable_welfare.get('group'):
            group_data = stable_welfare['group']
            print(f"  Group stable profiles: {group_data['count']}")
            if group_data['welfare_values']:
                print(f"    Best Group welfare: {group_data['max_welfare']:.4f}")
                print(f"    Worst Group welfare: {group_data['min_welfare']:.4f}")
        
        if stable_welfare.get('protective'):
            protective_data = stable_welfare['protective']
            print(f"  Protective stable profiles: {protective_data['count']}")
            if protective_data['welfare_values']:
                print(f"    Best Protective welfare: {protective_data['max_welfare']:.4f}")
                print(f"    Worst Protective welfare: {protective_data['min_welfare']:.4f}")
    
    def analyze_project_allocation_efficiency(self):
        """Analyze how project allocations affect social welfare."""
        print(f"\n" + "="*70)
        print("PROJECT ALLOCATION EFFICIENCY ANALYSIS")
        print("="*70)
        
        optimal_profiles, optimal_welfare = self.find_social_optimum()
        
        print(f"Optimal Project Allocations:")
        for i, profile in enumerate(optimal_profiles, 1):
            print(f"\nOptimal allocation {i}: {profile}")
            
            # Analyze each project's contribution
            project_contributions = {}
            for project in self.projects:
                coalition = self.get_coalition_for_project(profile, project)
                if coalition:
                    project_welfare = 0
                    for player in coalition:
                        utility = self.compute_utility(profile, player)
                        project_welfare += utility
                    project_contributions[project] = {
                        'coalition_size': len(coalition),
                        'coalition_members': set(coalition),
                        'total_contribution': project_welfare,
                        'per_player_avg': project_welfare / len(coalition) if coalition else 0
                    }
                else:
                    project_contributions[project] = {
                        'coalition_size': 0,
                        'coalition_members': set(),
                        'total_contribution': 0,
                        'per_player_avg': 0
                    }
            
            # Show project contributions
            print(f"  Project contributions to social welfare:")
            for project, contrib in project_contributions.items():
                print(f"    Project {project}:")
                print(f"      Coalition: {contrib['coalition_members']} (size: {contrib['coalition_size']})")
                print(f"      Total welfare contribution: {contrib['total_contribution']:.4f}")
                print(f"      Average per player: {contrib['per_player_avg']:.4f}")
        
        # Compare with worst-case allocation
        welfare_analysis = self.analyze_all_profiles_welfare()
        worst_profile, worst_welfare, _ = welfare_analysis[-1]
        
        print(f"\nComparison with worst allocation:")
        print(f"  Worst profile: {worst_profile}")
        print(f"  Worst welfare: {worst_welfare:.4f}")
        print(f"  Welfare gap: {optimal_welfare - worst_welfare:.4f}")
        print(f"  Relative improvement: {(optimal_welfare / worst_welfare - 1) * 100:.2f}%" if worst_welfare > 0 else "Infinite improvement")


def create_social_welfare_example_1():
    """Create Example 1 as a Social Welfare game."""
    players = [1, 2, 3]
    projects = ['a', 'b']
    rewards = {'a': 10, 'b': 16}
    
    preferences = {
        1: {
            frozenset([1]): 0.2,
            frozenset([1, 2]): 0.3,
            frozenset([1, 3]): 0.2,
            frozenset([1, 2, 3]): 0.3
        },
        2: {
            frozenset([2]): 0.1,
            frozenset([1, 2]): 0.4,
            frozenset([2, 3]): 0.2,
            frozenset([1, 2, 3]): 0.3
        },
        3: {
            frozenset([3]): 0.2,
            frozenset([1, 3]): 0.2,
            frozenset([2, 3]): 0.4,
            frozenset([1, 2, 3]): 0.2
        }
    }
    
    return SocialWelfare(players, projects, rewards, preferences)


def create_social_welfare_monotonic(preference_type: str = "harmonic"):
    """Create a monotonic decreasing game for social welfare analysis."""
    try:
        from MonotonicDecreasingProjectGame import MonotonicDecreasingProjectGame
        monotonic_game = MonotonicDecreasingProjectGame([1, 2, 3], ['a', 'b'], {'a': 10, 'b': 16}, preference_type)
        return SocialWelfare(monotonic_game.players, monotonic_game.projects, 
                           monotonic_game.rewards, monotonic_game.preferences)
    except ImportError:
        print("MonotonicDecreasingProjectGame not available, using regular example")
        return create_social_welfare_example_1()


def main():
    """Demonstrate social welfare analysis."""
    print("=" * 70)
    print("SOCIAL WELFARE AND SOCIAL OPTIMUM IN PROJECT HEDONIC GAMES")
    print("=" * 70)
    
    print("Mathematical Definitions:")
    print("  Social Welfare: SW(s) = Σ_{i ∈ N} u_i(s)")
    print("  Social Optimum: s* = argmax_{s ∈ S} SW(s)")
    print("  Optimal Value: OPT(G) = SW(s*)")
    
    # Analyze Example 1
    print(f"\n" + "="*70)
    print("EXAMPLE 1 ANALYSIS")
    print("="*70)
    
    game1 = create_social_welfare_example_1()
    game1.print_game_info()
    game1.print_social_welfare_analysis()
    game1.analyze_project_allocation_efficiency()
    
    # Analyze Monotonic Decreasing Game
    print(f"\n" + "="*70)
    print("MONOTONIC DECREASING PREFERENCES ANALYSIS")
    print("="*70)
    
    game2 = create_social_welfare_monotonic("harmonic")
    print("Using harmonic monotonic decreasing preferences...")
    game2.print_social_welfare_analysis()
    game2.analyze_project_allocation_efficiency()


def demo_basic():
    """Basic demonstration for testing."""
    print("Basic Social Welfare Demo")
    print("=" * 30)
    
    game = create_social_welfare_example_1()
    print(f"Created game with {len(game.players)} players and {len(game.projects)} projects")
    
    # Test social welfare computation
    test_profile = {1: 'a', 2: 'a', 3: 'a'}
    welfare = game.compute_social_welfare(test_profile)
    print(f"\nSocial welfare for profile {test_profile}: {welfare:.4f}")
    
    # Find social optimum
    optimal_profiles, optimal_welfare = game.find_social_optimum()
    print(f"\nSocial optimum:")
    print(f"  Optimal welfare: {optimal_welfare:.4f}")
    print(f"  Number of optimal profiles: {len(optimal_profiles)}")
    print(f"  Optimal profiles: {optimal_profiles}")


if __name__ == "__main__":
    main()