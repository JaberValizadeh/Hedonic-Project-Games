import math
from itertools import product, combinations
from typing import Dict, List, Set, Tuple, Any, Callable
from HedonicProjectGame import HedonicProjectGame


class MonotonicDecreasingProjectGame(HedonicProjectGame):
    """
    Implementation of Hedonic Project Games with Monotonic Decreasing Preferences.
    
    This class implements Hedonic Project Games where preferences satisfy:
    ∀ i ∈ N, ∀ C, C' ∈ N_i: p_i(C) ≥ p_i(C') ⟺ |C| ≤ |C'|
    
    In other words, each player's preference for a coalition depends only on the coalition size
    and decreases monotonically as the coalition size increases. Players prefer smaller coalitions.
    
    Mathematical Framework:
    ----------------------
    Let N_i = {C ∈ 2^N | i ∈ C} denote the set of possible coalitions that player i belongs to,
    where 2^N is the power set of N. Each player i has a preference function p_i: N_i → [0,1]
    that satisfies the normalization constraint:
    
        ∑_{C ∈ N_i} p_i(C) = 1
    
    For monotonic decreasing preferences, all coalitions of the same size have identical 
    preference values, so we can express preferences as p_i(|C|) where |C| is coalition size.
    
    The normalization constraint becomes:
        ∑_{k=1}^n C(n-1, k-1) × p_i(k) = 1
    
    where C(n-1, k-1) is the number of coalitions of size k containing player i,
    and n = |N| is the total number of players.
    
    This implementation automatically computes normalized preference values for any
    monotonic decreasing base function, ensuring the mathematical constraint is satisfied.
    """
    
    def __init__(self, players: List[int], projects: List[Any], rewards: Dict[Any, float],
                 preference_type: str = "harmonic"):
        """
        Initialize a Monotonic Decreasing Preferences Project Hedonic Game.
        
        Args:
            players: List of player IDs (N)
            projects: List of project IDs (M)
            rewards: Dictionary mapping project to reward value (r function)
            preference_type: Type of monotonic decreasing function to use:
                - "harmonic": p_i(|C|) = 1/|C|
                - "exponential": p_i(|C|) = exp(-α*|C|) where α > 0
                - "linear": p_i(|C|) = max(0, 1 - α*|C|) where α > 0
                - "reciprocal": p_i(|C|) = 1/|C|² 
                - "logarithmic": p_i(|C|) = 1/log(|C|+1)
        """
        # Generate monotonic decreasing preferences automatically
        preferences = self._generate_monotonic_preferences(players, preference_type)
        
        # Initialize the parent class
        super().__init__(players, projects, rewards, preferences)
        
        self.preference_type = preference_type
        
        # Validate that preferences are indeed monotonic decreasing
        self._validate_monotonic_decreasing_property()
        
        # Validate that preferences satisfy the normalization constraint ∑_{C ∈ N_i} p_i(C) = 1
        self._validate_preference_normalization()
    
    def _generate_monotonic_preferences(self, players: List[int], 
                                      preference_type: str) -> Dict[int, Dict[frozenset, float]]:
        """
        Generate monotonic decreasing preferences for all players.
        
        Since preferences depend only on coalition size, all coalitions of the same size
        containing a player have the same preference value.
        
        Mathematical Process:
        --------------------
        1. Generate raw preference values g(k) for each coalition size k
        2. Compute normalization factor: Z = ∑_{k=1}^n C(n-1, k-1) × g(k)
        3. Set normalized preferences: p_i(k) = g(k) / Z
        4. Verify: ∑_{k=1}^n C(n-1, k-1) × p_i(k) = 1
        
        where C(n-1, k-1) = (n-1)! / ((k-1)! × (n-k)!) is the number of coalitions
        of size k containing a specific player i.
        
        Args:
            players: List of player IDs
            preference_type: Type of monotonic function to use
            
        Returns:
            Dictionary mapping player to their preference function over coalitions
        """
        preferences = {}
        n = len(players)
        
        # Get the base preference function g(k)
        pref_func = self._get_preference_function(preference_type, n)
        
        # Calculate raw preference values for each coalition size
        size_preferences = {}
        for size in range(1, n + 1):
            size_preferences[size] = pref_func(size)
        
        # Normalize to satisfy ∑_{C ∈ N_i} p_i(C) = 1
        # Compute normalization factor Z = ∑_{k=1}^n C(n-1, k-1) × g(k)
        total_weighted_pref = 0
        for size in range(1, n + 1):
            # Number of coalitions of size 'size' containing a specific player
            num_coalitions = math.comb(n - 1, size - 1)
            total_weighted_pref += num_coalitions * size_preferences[size]
        
        # Apply normalization: p_i(k) = g(k) / Z
        for size in size_preferences:
            size_preferences[size] /= total_weighted_pref
        
        # Store size-based preferences for efficient access
        self._size_preferences = size_preferences
        
        # Generate preferences for all players
        for player in players:
            player_prefs = {}
            
            # Generate all possible coalitions containing this player
            for coalition_size in range(1, n + 1):
                for coalition_members in combinations(players, coalition_size):
                    if player in coalition_members:
                        coalition = frozenset(coalition_members)
                        player_prefs[coalition] = size_preferences[coalition_size]
            
            preferences[player] = player_prefs
        
        return preferences
    
    def _get_preference_function(self, preference_type: str, n: int) -> Callable[[int], float]:
        """
        Get the base preference function for a given type.
        
        Args:
            preference_type: Type of monotonic function
            n: Number of players (for normalization)
            
        Returns:
            Function that maps coalition size to preference value
        """
        if preference_type == "harmonic":
            return lambda size: 1.0 / size
        
        elif preference_type == "exponential":
            # Use α = 0.5 for moderate exponential decay
            alpha = 0.5
            return lambda size: math.exp(-alpha * size)
        
        elif preference_type == "linear":
            # Use α such that preference becomes 0 at maximum coalition size
            alpha = 1.0 / (n + 1)
            return lambda size: max(0.0, 1.0 - alpha * size)
        
        elif preference_type == "reciprocal":
            return lambda size: 1.0 / (size * size)
        
        elif preference_type == "logarithmic":
            return lambda size: 1.0 / math.log(size + 1)
        
        else:
            raise ValueError(f"Unknown preference type: {preference_type}")
    
    def _validate_monotonic_decreasing_property(self):
        """
        Validate that the generated preferences satisfy the monotonic decreasing property.
        
        For each player i and coalitions C, C' ∈ N_i:
        p_i(C) ≥ p_i(C') ⟺ |C| ≤ |C'|
        """
        for player in self.players:
            player_prefs = self.preferences[player]
            
            # Group coalitions by size and check that all coalitions of same size have same preference
            size_to_pref = {}
            for coalition, pref_value in player_prefs.items():
                size = len(coalition)
                if size in size_to_pref:
                    if abs(size_to_pref[size] - pref_value) > 1e-10:
                        raise ValueError(
                            f"Player {player}: Coalitions of size {size} have different preferences: "
                            f"{size_to_pref[size]:.6f} vs {pref_value:.6f}"
                        )
                else:
                    size_to_pref[size] = pref_value
            
            # Check that preferences decrease with coalition size
            sizes = sorted(size_to_pref.keys())
            for i in range(len(sizes) - 1):
                size1, size2 = sizes[i], sizes[i + 1]
                pref1, pref2 = size_to_pref[size1], size_to_pref[size2]
                
                if pref1 < pref2 - 1e-10:  # Allow for small numerical errors
                    raise ValueError(
                        f"Player {player}: Monotonic decreasing property violated. "
                        f"Size {size1} has preference {pref1:.6f} < "
                        f"Size {size2} has preference {pref2:.6f}"
                    )
    
    def _validate_preference_normalization(self):
        """
        Validate that preferences satisfy the mathematical constraint from the paper:
        ∑_{C ∈ N_i} p_i(C) = 1 for each player i ∈ N
        
        Where N_i = {C ∈ 2^N | i ∈ C} is the set of all coalitions containing player i.
        This ensures that p_i: N_i → [0,1] is a proper probability distribution.
        """
        for player in self.players:
            player_prefs = self.preferences[player]
            total_preference = sum(player_prefs.values())
            
            if abs(total_preference - 1.0) > 1e-10:
                raise ValueError(
                    f"Normalization constraint violated for player {player}: "
                    f"∑_{{C ∈ N_{player}}} p_{player}(C) = {total_preference:.10f} ≠ 1.0 "
                    f"(difference: {abs(total_preference - 1.0):.2e})"
                )
    
    def get_preference_by_coalition_size(self, player: int, coalition_size: int) -> float:
        """
        Get the preference value for any coalition of a given size containing the player.
        
        Since preferences are monotonic decreasing, all coalitions of the same size
        containing the player have the same preference value.
        
        Args:
            player: Player ID
            coalition_size: Size of the coalition
            
        Returns:
            Preference value for coalitions of this size
        """
        if player not in self.players:
            raise ValueError(f"Player {player} not in game")
        
        if coalition_size < 1 or coalition_size > len(self.players):
            raise ValueError(f"Invalid coalition size: {coalition_size}")
        
        # Use cached size preferences for efficiency
        return self._size_preferences[coalition_size]
    
    def check_normalization_constraint(self, verbose: bool = True) -> bool:
        """
        Check that the mathematical constraint ∑_{C ∈ N_i} p_i(C) = 1 holds for all players.
        
        This verifies that each player's preference function p_i: N_i → [0,1] satisfies
        the normalization constraint from the paper, where N_i = {C ∈ 2^N | i ∈ C}.
        
        Args:
            verbose: Whether to print detailed results
            
        Returns:
            True if constraint is satisfied for all players, False otherwise
        """
        try:
            if verbose:
                print("Checking normalization constraint: ∑_{C ∈ N_i} p_i(C) = 1")
                print("=" * 60)
            
            self._validate_preference_normalization()
            
            if verbose:
                print("✓ All players satisfy the normalization constraint!")
                
                # Show mathematical breakdown
                n = len(self.players)
                print(f"\nMathematical verification for n = {n} players:")
                print("-" * 40)
                
                for size in sorted(self._size_preferences.keys()):
                    pref_value = self._size_preferences[size]
                    num_coalitions = math.comb(n - 1, size - 1)
                    contribution = num_coalitions * pref_value
                    
                    print(f"  Size {size}: C({n-1},{size-1}) × p_i(|C|={size}) = "
                          f"{num_coalitions} × {pref_value:.6f} = {contribution:.6f}")
                
                total = sum(math.comb(n - 1, size - 1) * self._size_preferences[size] 
                           for size in range(1, n + 1))
                print(f"  Total: {total:.10f}")
            
            return True
            
        except ValueError as e:
            if verbose:
                print(f"✗ Normalization constraint violated: {e}")
            return False
    
    def compute_utility_efficient(self, player: int, strategy_profile: Dict[int, Any]) -> float:
        """
        Efficiently compute utility for a player given a strategy profile.
        
        This method uses the fact that preferences depend only on coalition size
        to avoid expensive coalition enumeration.
        
        Args:
            player: Player ID
            strategy_profile: Dictionary mapping players to their chosen projects
            
        Returns:
            Utility value for the player
        """
        if player not in self.players:
            raise ValueError(f"Player {player} not in game")
        
        player_project = strategy_profile[player]
        
        # Find all players choosing the same project (coalition members)
        coalition_members = [p for p, proj in strategy_profile.items() if proj == player_project]
        coalition_size = len(coalition_members)
        
        # Get preference based on coalition size
        preference = self.get_preference_by_coalition_size(player, coalition_size)
        
        # Get project reward
        project_reward = self.rewards[player_project]
        
        # Utility = preference × (reward / coalition_size) following Equation (1)
        return preference * (project_reward / coalition_size)
    
    def get_all_strategy_profiles(self) -> List[Dict[int, Any]]:
        """
        Generate all possible strategy profiles.
        
        Returns:
            List of all possible strategy profiles where each profile maps players to projects
        """
        # Generate all combinations where each player chooses a project
        player_project_choices = []
        for player in self.players:
            player_project_choices.append([(player, project) for project in self.projects])
        
        # Generate all possible strategy profiles
        all_profiles = []
        for profile_tuple in product(*player_project_choices):
            profile = {}
            for player, project in profile_tuple:
                profile[player] = project
            all_profiles.append(profile)
        
        return all_profiles
    
    def compute_social_welfare(self, strategy_profile: Dict[int, Any]) -> float:
        """
        Compute social welfare for a given strategy profile.
        
        Args:
            strategy_profile: Dictionary mapping players to their chosen projects
            
        Returns:
            Sum of all players' utilities
        """
        utilities = self.compute_all_utilities(strategy_profile)
        return sum(utilities.values())
    
    def display_all_strategy_profiles_table(self):
        """
        Display a comprehensive table showing all strategy profiles with utilities and social welfare.
        """
        print(f"Utility analysis for all profiles:")
        
        # Get all strategy profiles
        all_profiles = self.get_all_strategy_profiles()
        
        # Create header based on number of players
        if len(self.players) == 3:
            print(f"Profile      | u_1(.) | u_2(.) | u_3(.) | SW(.)")
            print(f"-------------|--------|--------|--------|--------")
        elif len(self.players) == 4:
            print(f"Profile      | u_1(.) | u_2(.) | u_3(.) | u_4(.) | SW(.)")
            print(f"-------------|--------|--------|--------|--------|--------")
        else:
            # Dynamic header for other numbers of players
            header = "Profile      |"
            separator = "-------------|"
            for player in sorted(self.players):
                header += f" u_{player}(.) |"
                separator += "--------|"
            header += " SW(.)"
            separator += "--------"
            print(header)
            print(separator)
        
        # Print each strategy profile
        for idx, profile in enumerate(all_profiles, 1):
            # Compute utilities and social welfare
            utilities = self.compute_all_utilities(profile)
            social_welfare = sum(utilities.values())
            
            # Create profile description
            profile_desc = "("
            for i, player in enumerate(sorted(self.players)):
                if i > 0:
                    profile_desc += ","
                profile_desc += f"{profile[player]}"
            profile_desc += ")"
            
            # Create row
            row = f"{profile_desc:<12} |"
            
            # Add utilities
            for player in sorted(self.players):
                row += f" {utilities[player]:6.4f} |"
            
            # Add social welfare
            row += f" {social_welfare:6.4f}"
            
            print(row)
        
        # Add summary statistics
        all_social_welfares = []
        for profile in all_profiles:
            utilities = self.compute_all_utilities(profile)
            all_social_welfares.append(sum(utilities.values()))
        
        max_sw = max(all_social_welfares)
        min_sw = min(all_social_welfares)
        avg_sw = sum(all_social_welfares) / len(all_social_welfares)
        
        print(f"\nSUMMARY STATISTICS:")
        print(f"Total strategy profiles: {len(all_profiles)}")
        print(f"Maximum social welfare: {max_sw:.4f}")
        print(f"Minimum social welfare: {min_sw:.4f}")
        print(f"Average social welfare: {avg_sw:.4f}")
        
        # Find optimal strategy profiles
        optimal_profiles = [profile for profile in all_profiles 
                          if abs(self.compute_social_welfare(profile) - max_sw) < 1e-10]
        
        print(f"\nOptimal strategy profile(s) (SW = {max_sw:.4f}):")
        for i, profile in enumerate(optimal_profiles, 1):
            profile_str = ", ".join([f"P{player}→{project}" for player, project in sorted(profile.items())])
            print(f"  {i}. {profile_str}")
    
    def print_preference_structure(self):
        """Print the preference structure showing monotonic decreasing property."""
        print(f"Monotonic Decreasing Preferences ({self.preference_type}):")
        print("=" * 60)
        
        # Show size-based preference values (same for all players)
        print("\nPreference by Coalition Size (same for all players):")
        print("-" * 50)
        for size in sorted(self._size_preferences.keys()):
            pref_value = self._size_preferences[size]
            # Count how many coalitions of this size exist for each player
            num_coalitions = math.comb(len(self.players) - 1, size - 1)
            print(f"  Size {size}: p_i(|C| = {size}) = {pref_value:.6f} "
                  f"({num_coalitions} coalition{'s' if num_coalitions > 1 else ''} per player)")
        
        # Verify monotonic decreasing property
        sizes = sorted(self._size_preferences.keys())
        print(f"\nMonotonic Decreasing Property Verification:")
        print("-" * 40)
        for i in range(len(sizes) - 1):
            size1, size2 = sizes[i], sizes[i + 1]
            pref1, pref2 = self._size_preferences[size1], self._size_preferences[size2]
            ratio = pref1 / pref2 if pref2 > 0 else float('inf')
            print(f"  p_i(|C| = {size1}) / p_i(|C| = {size2}) = {pref1:.6f} / {pref2:.6f} = {ratio:.3f}")
        
        # Show sample coalitions for one player
        if self.players:
            sample_player = self.players[0]
            print(f"\nSample Coalitions for Player {sample_player}:")
            print("-" * 40)
            
            size_to_coalitions = {}
            for coalition, pref_value in self.preferences[sample_player].items():
                size = len(coalition)
                if size not in size_to_coalitions:
                    size_to_coalitions[size] = []
                size_to_coalitions[size].append(coalition)
            
            for size in sorted(size_to_coalitions.keys()):
                coalitions = size_to_coalitions[size]
                pref_value = self._size_preferences[size]
                
                print(f"  Size {size} (p = {pref_value:.6f}):")
                
                # Show a few example coalitions
                if len(coalitions) <= 3:
                    for coalition in coalitions:
                        print(f"    {set(coalition)}")
                else:
                    for coalition in coalitions[:2]:
                        print(f"    {set(coalition)}")
                    print(f"    ... and {len(coalitions) - 2} more")


def create_monotonic_example_1(preference_type: str = "harmonic"):
    """
    Create Example 1 from the paper with monotonic decreasing preferences.
    
    Args:
        preference_type: Type of monotonic decreasing function to use
        
    Returns:
        MonotonicDecreasingProjectGame instance
    """
    players = [1, 2, 3]
    projects = ['a', 'b']
    rewards = {'a': 10, 'b': 16}
    
    return MonotonicDecreasingProjectGame(players, projects, rewards, preference_type)


def create_monotonic_example_2(preference_type: str = "exponential"):
    """
    Create a larger example with monotonic decreasing preferences.
    
    Args:
        preference_type: Type of monotonic decreasing function to use
        
    Returns:
        MonotonicDecreasingProjectGame instance
    """
    players = [1, 2, 3, 4]
    projects = ['a', 'b', 'c']
    rewards = {'a': 8, 'b': 12, 'c': 20}
    
    return MonotonicDecreasingProjectGame(players, projects, rewards, preference_type)


def main():
    """Demonstrate monotonic decreasing hedonic project games with comprehensive analysis."""
    print("Monotonic Decreasing Hedonic Project Games")
    print("=" * 50)
    
    # Create a game instance with harmonic preferences for detailed analysis
    print("\nCreating game with HARMONIC preferences for detailed analysis:")
    print("-" * 60)
    
    game = create_monotonic_example_1("harmonic")
    
    # Show game setup
    print(f"Players: {game.players}")
    print(f"Projects: {game.projects}")
    print(f"Rewards: {game.rewards}")
    
    # Show preference structure
    game.print_preference_structure()
    
    # Display comprehensive strategy profiles table
    game.display_all_strategy_profiles_table()
    
    # Test normalization constraint
    print(f"\n" + "=" * 60)
    print("MATHEMATICAL CONSTRAINT VERIFICATION")
    print("=" * 60)
    game.check_normalization_constraint(verbose=True)
    
    # Additional analysis for other preference types (more concise)
    print(f"\n\n" + "=" * 80)
    print("COMPARISON ACROSS DIFFERENT PREFERENCE TYPES")
    print("=" * 80)
    
    preference_types = ["exponential", "linear", "reciprocal", "logarithmic"]
    
    for pref_type in preference_types:
        print(f"\n{pref_type.upper()} Preferences:")
        print("-" * 40)
        
        try:
            test_game = create_monotonic_example_1(pref_type)
            
            # Show just the strategy table for comparison
            test_game.display_all_strategy_profiles_table()
            
        except Exception as e:
            print(f"Error with {pref_type} preferences: {e}")
    
    print(f"\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
