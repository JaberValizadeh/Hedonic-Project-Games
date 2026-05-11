import math
import random
from itertools import product, combinations
from typing import Dict, List, Set, Tuple, Any, Optional
from HedonicProjectGame import HedonicProjectGame


class AdditivitySeparableGame(HedonicProjectGame):
    """
    Implementation of Hedonic Project Games with Additivity Separable Preferences and Symmetry.
    
    Mathematical Definition:
    - Agent i's preference for coalition C ∈ N_i is: p_i(C) = Σ_{i' ∈ C\\{i}} w_i(i')
    - w_i: N → [0, 1] is the value function of agent i assigning weight w_i(i') ∈ [0, 1] to each agent i' ∈ N
    - Symmetry property: w_i(i') = w_{i'}(i) for all i, i' ∈ N
    - Normalization condition: p_i: N_i → [0,1] where Σ_{C ∈ N_i} p_i(C) = 1
    
    Key Properties:
    1. Additivity: Preference is sum of individual agent weights
    2. Separability: Each agent contributes independently to coalition value  
    3. Symmetry: Mutual valuation between any pair of agents
    4. Normalization: Preferences form a proper probability distribution over coalitions
    
    Implementation Details:
    - Raw additivity separable preferences are computed first using the formula above
    - Preferences are then normalized to satisfy Σ_{C ∈ N_i} p_i(C) = 1
    - Both raw and normalized preferences are stored for analysis
    - Normalization preserves relative preference orderings
    """
    
    def __init__(self, players: List[int], projects: List[Any], rewards: Dict[Any, float],
                 weight_matrix: Optional[Dict[Tuple[int, int], float]] = None,
                 weight_generation_method: str = "random_symmetric"):
        """
        Initialize an Additivity Separable Project Hedonic Game.
        
        Args:
            players: List of player IDs (N)
            projects: List of project IDs (M)
            rewards: Dictionary mapping project to reward value (r function)
            weight_matrix: Optional pre-specified weight matrix as {(i, i'): w_i(i')}
            weight_generation_method: Method for generating weights if not provided
                - "random_symmetric": Random weights with symmetry enforcement
                - "distance_based": Weights based on player ID distances
                - "uniform": All weights equal to 1/(n-1) where n = |N|
        """
        self.weight_generation_method = weight_generation_method
        
        # Generate or validate weight matrix
        if weight_matrix is None:
            self.weight_matrix = self._generate_symmetric_weights(players, weight_generation_method)
        else:
            self.weight_matrix = weight_matrix
            self._validate_symmetry(players)
        
        # Generate additivity separable preferences
        self.raw_preferences = self._generate_additivity_separable_preferences(players)
        
        # Normalize preferences to sum to 1 for each player (required by parent class)
        normalized_preferences = self._normalize_preferences(self.raw_preferences, players)
        
        # Initialize parent class
        super().__init__(players, projects, rewards, normalized_preferences)
        
        # Validate additivity separable property
        self._validate_additivity_separable_property()
        
        # Validate symmetry property  
        self._validate_symmetry_property()
        
        # Validate normalization condition: Σ_{C ∈ N_i} p_i(C) = 1
        self._validate_preference_normalization()
    
    def _generate_symmetric_weights(self, players: List[int], method: str) -> Dict[Tuple[int, int], float]:
        """
        Generate symmetric weight matrix w_i(i') = w_{i'}(i).
        
        Args:
            players: List of player IDs
            method: Weight generation method
            
        Returns:
            Dictionary mapping (i, i') to weight w_i(i')
        """
        weight_matrix = {}
        n = len(players)
        
        if method == "random_symmetric":
            # Generate random symmetric weights
            for i in range(len(players)):
                for j in range(i + 1, len(players)):
                    player_i, player_j = players[i], players[j]
                    # Generate random weight between 0 and 1
                    weight = random.uniform(0.1, 1.0)  # Avoid 0 to ensure some preference
                    
                    # Set symmetric weights
                    weight_matrix[(player_i, player_j)] = weight
                    weight_matrix[(player_j, player_i)] = weight
            
            # Self-weights are 0 (agent doesn't value themselves in additivity)
            for player in players:
                weight_matrix[(player, player)] = 0.0
                
        elif method == "distance_based":
            # Weights based on inverse of ID distance
            for i in range(len(players)):
                for j in range(len(players)):
                    player_i, player_j = players[i], players[j]
                    if i == j:
                        weight_matrix[(player_i, player_j)] = 0.0
                    else:
                        # Inverse distance normalized
                        distance = abs(player_i - player_j)
                        weight = 1.0 / (1.0 + distance)
                        weight_matrix[(player_i, player_j)] = weight
                        
        elif method == "uniform":
            # All non-self weights equal
            uniform_weight = 1.0 / max(1, n - 1)  # Avoid division by zero
            for i in range(len(players)):
                for j in range(len(players)):
                    player_i, player_j = players[i], players[j]
                    if i == j:
                        weight_matrix[(player_i, player_j)] = 0.0
                    else:
                        weight_matrix[(player_i, player_j)] = uniform_weight
        
        return weight_matrix
    
    def _generate_additivity_separable_preferences(self, players: List[int]) -> Dict[int, Dict[frozenset, float]]:
        """
        Generate preferences using additivity separable formula: p_i(C) = Σ_{i' ∈ C\\{i}} w_i(i').
        
        Args:
            players: List of player IDs
            
        Returns:
            Dictionary mapping player to their preferences over coalitions
        """
        preferences = {}
        
        for player in players:
            player_preferences = {}
            
            # Generate all possible coalitions containing this player
            other_players = [p for p in players if p != player]
            
            # Coalition of size 1 (just the player)
            singleton_coalition = frozenset([player])
            player_preferences[singleton_coalition] = 0.0  # No other agents to value
            
            # Coalitions of size 2 and above
            for size in range(2, len(players) + 1):
                for other_subset in combinations(other_players, size - 1):
                    coalition = frozenset([player] + list(other_subset))
                    
                    # Calculate additivity separable preference: Σ_{i' ∈ C\{i}} w_i(i')
                    preference_value = 0.0
                    for other_player in other_subset:
                        preference_value += self.weight_matrix[(player, other_player)]
                    
                    player_preferences[coalition] = preference_value
            
            preferences[player] = player_preferences
        
        return preferences
    
    def _normalize_preferences(self, preferences: Dict[int, Dict[frozenset, float]], 
                              players: List[int]) -> Dict[int, Dict[frozenset, float]]:
        """
        Normalize additivity separable preferences to sum to 1 for each player.
        
        Args:
            preferences: Raw additivity separable preferences
            players: List of player IDs
            
        Returns:
            Normalized preferences where each player's preferences sum to 1
        """
        normalized_preferences = {}
        
        for player in players:
            player_prefs = preferences[player]
            total = sum(player_prefs.values())
            
            if total > 0:
                # Normalize by dividing by total
                normalized_preferences[player] = {
                    coalition: pref / total for coalition, pref in player_prefs.items()
                }
            else:
                # If all preferences are 0, set equal probabilities
                num_coalitions = len(player_prefs)
                normalized_preferences[player] = {
                    coalition: 1.0 / num_coalitions for coalition in player_prefs.keys()
                }
        
        return normalized_preferences
    
    def _validate_preference_normalization(self):
        """
        Validate that preferences are properly normalized: Σ_{C ∈ N_i} p_i(C) = 1 for each agent i.
        
        This ensures the mathematical constraint that each agent's preference function 
        p_i: N_i → [0,1] satisfies the normalization condition.
        """
        for player in self.players:
            player_prefs = self.preferences[player]
            total_preference = sum(player_prefs.values())
            
            if abs(total_preference - 1.0) > 1e-10:
                raise ValueError(
                    f"Normalization violation for player {player}: "
                    f"Σ_{{C ∈ N_{player}}} p_{player}(C) = {total_preference:.10f} ≠ 1.0 "
                    f"(difference: {abs(total_preference - 1.0):.2e})"
                )
            
            # Validation passed silently
    
    def check_preference_normalization(self, verbose: bool = True) -> bool:
        """
        Check if all players' preferences are properly normalized (sum to 1).
        
        Mathematical constraint: For each agent i, Σ_{C ∈ N_i} p_i(C) = 1
        where p_i: N_i → [0,1] is agent i's preference function.
        
        Args:
            verbose: If True, print detailed information about each player's preferences
            
        Returns:
            True if all preferences are normalized, False otherwise
        """
        all_normalized = True
        
        if verbose:
            print(f"\nPREFERENCE NORMALIZATION CHECK")
            print("Mathematical constraint: Σ_{C ∈ N_i} p_i(C) = 1 for each agent i")
            print("-" * 70)
        
        for player in self.players:
            player_prefs = self.preferences[player]
            total_preference = sum(player_prefs.values())
            is_normalized = abs(total_preference - 1.0) < 1e-10
            
            if verbose:
                print(f"Player {player}:")
                print(f"  Total preference sum: {total_preference:.10f}")
                print(f"  Normalized: {'[OK] YES' if is_normalized else '[FAIL] NO'}")
                
                if not is_normalized:
                    print(f"  ERROR: Expected Σ_{{C ∈ N_{player}}} p_{player}(C) = 1.0, actual sum = {total_preference:.10f}")
                    print(f"  Difference: {abs(total_preference - 1.0):.2e}")
            
            if not is_normalized:
                all_normalized = False
        
        if verbose:
            print(f"\nOverall result: {'[OK] ALL NORMALIZED' if all_normalized else '[FAIL] NORMALIZATION FAILED'}")
        
        return all_normalized
    
    def _validate_symmetry(self, players: List[int]):
        """Validate that weight matrix satisfies symmetry: w_i(i') = w_{i'}(i)."""
        for i in players:
            for j in players:
                if (i, j) not in self.weight_matrix or (j, i) not in self.weight_matrix:
                    raise ValueError(f"Missing weight entries for players {i} and {j}")
                
                if abs(self.weight_matrix[(i, j)] - self.weight_matrix[(j, i)]) > 1e-10:
                    raise ValueError(f"Symmetry violation: w_{i}({j}) = {self.weight_matrix[(i, j)]:.6f} "
                                   f"but w_{j}({i}) = {self.weight_matrix[(j, i)]:.6f}")
    
    def _validate_additivity_separable_property(self):
        """Validate that raw preferences satisfy additivity separable property."""
        for player in self.players:
            for coalition, preference in self.raw_preferences[player].items():
                if player not in coalition:
                    continue
                
                # Calculate expected preference using additivity formula
                expected_preference = 0.0
                for other_player in coalition:
                    if other_player != player:
                        expected_preference += self.weight_matrix[(player, other_player)]
                
                if abs(preference - expected_preference) > 1e-10:
                    raise ValueError(f"Additivity violation for player {player}, coalition {set(coalition)}: "
                                   f"expected {expected_preference:.6f}, got {preference:.6f}")
    
    def _validate_symmetry_property(self):
        """Validate that the overall preference structure satisfies symmetry."""
        for i in self.players:
            for j in self.players:
                if i != j:
                    weight_ij = self.weight_matrix[(i, j)]
                    weight_ji = self.weight_matrix[(j, i)]
                    
                    if abs(weight_ij - weight_ji) > 1e-10:
                        raise ValueError(f"Symmetry property violated: w_{i}({j}) ≠ w_{j}({i})")
    
    def get_weight(self, player_i: int, player_j: int) -> float:
        """
        Get weight w_i(j) that player i assigns to player j.
        
        Args:
            player_i: The player assigning the weight
            player_j: The player being valued
            
        Returns:
            Weight value w_i(j)
        """
        return self.weight_matrix.get((player_i, player_j), 0.0)
    
    def get_weight_matrix_display(self) -> str:
        """
        Get a formatted display of the weight matrix.
        
        Returns:
            Formatted string showing the weight matrix
        """
        result = "Weight Matrix (w_i(j) where i=row, j=column):\n"
        result += "     "
        
        # Header row
        for j in sorted(self.players):
            result += f"{j:8}"
        result += "\n"
        
        # Data rows
        for i in sorted(self.players):
            result += f"{i:4} "
            for j in sorted(self.players):
                weight = self.weight_matrix[(i, j)]
                result += f"{weight:8.4f}"
            result += "\n"
        
        return result
    
    def analyze_coalition_preferences(self, player: int) -> Dict[frozenset, Tuple[float, List[float]]]:
        """
        Analyze how coalition preferences are built from individual weights.
        
        Args:
            player: Player to analyze
            
        Returns:
            Dictionary mapping coalition to (raw_preference, individual_contributions)
        """
        analysis = {}
        
        for coalition, total_pref in self.raw_preferences[player].items():
            if player not in coalition:
                continue
                
            individual_contributions = []
            for other_player in sorted(coalition):
                if other_player != player:
                    contribution = self.weight_matrix[(player, other_player)]
                    individual_contributions.append(contribution)
            
            analysis[coalition] = (total_pref, individual_contributions)
        
        return analysis
    
    def demonstrate_additivity_property(self, player: int):
        """
        Demonstrate the additivity separable property for a specific player.
        
        Args:
            player: Player to demonstrate for
        """
        print(f"\nAdditivity Separable Preferences for Player {player}")
        print("=" * 60)
        print(f"Formula: p_{player}(C) = Σ_{{i' ∈ C\\{{{player}}}}} w_{player}(i')")
        print()
        
        analysis = self.analyze_coalition_preferences(player)
        
        for coalition in sorted(analysis.keys(), key=lambda x: (len(x), sorted(list(x)))):
            total_pref, contributions = analysis[coalition]
            other_players = [p for p in sorted(coalition) if p != player]
            
            if not other_players:
                coalition_str = str(set(coalition))
                print(f"Coalition {coalition_str:15}: {total_pref:.4f} (singleton)")
            else:
                coalition_str = str(set(coalition))
                contrib_str = " + ".join([f"w_{player}({p})={c:.4f}" for p, c in zip(other_players, contributions)])
                print(f"Coalition {coalition_str:15}: {total_pref:.4f} = {contrib_str}")


def create_additivity_separable_example():
    """
    Create an example of additivity separable preferences with symmetry.
    """
    players = [1, 2, 3, 4]
    projects = ['a', 'b', 'c', 'd']
    rewards = {'a': 10, 'b': 5, 'c': 5, 'd': 5}
    
    # Create symmetric weight matrix
    weight_matrix = {
        (1, 1): 0.0, (1, 2): 0.4, (1, 3): 0.1, (1, 4): 0.1,
        (2, 1): 0.4, (2, 2): 0.0, (2, 3): 0.2, (2, 4): 0.1,
        (3, 1): 0.1, (3, 2): 0.2, (3, 3): 0.0, (3, 4): 0.1,
        (4, 1): 0.1, (4, 2): 0.1, (4, 3): 0.1, (4, 4): 0.0,
    }
    # players = [1, 2, 3, 4]
    # projects = ['a', 'b', 'c', 'd']
    # rewards = {'a': 100, 'b': 1, 'c': 1, 'd': 1}
    
    # # Create symmetric weight matrix
    # weight_matrix = {
    #     (1, 1): 0.0, (1, 2): 0.4, (1, 3): 0.1, (1, 4): 0.1,
    #     (2, 1): 0.4, (2, 2): 0.0, (2, 3): 0.2, (2, 4): 0.1,
    #     (3, 1): 0.1, (3, 2): 0.2, (3, 3): 0.0, (3, 4): 0.1,
    #     (4, 1): 0.1, (4, 2): 0.1, (4, 3): 0.1, (4, 4): 0.0,
    # }
    return AdditivitySeparableGame(players, projects, rewards, weight_matrix)


def main():
    """Demonstrate additivity separable preferences with symmetry."""
    print("Additivity Separable Preferences with Symmetry")
    print("=" * 60)
    
    # Create example game
    game = create_additivity_separable_example()
    
    # Display weight matrix
    print(game.get_weight_matrix_display())
    
    # Demonstrate additivity property for each player
    for player in game.players:
        game.demonstrate_additivity_property(player)
    
    # Verify symmetry
    print("\nSymmetry Verification:")
    print("=" * 30)
    for i in game.players:
        for j in game.players:
            if i < j:  # Avoid duplicate pairs
                w_ij = game.get_weight(i, j)
                w_ji = game.get_weight(j, i)
                print(f"w_{i}({j}) = {w_ij:.4f}, w_{j}({i}) = {w_ji:.4f} → {'✓' if abs(w_ij - w_ji) < 1e-10 else '✗'}")
    
    # Show all preferences
    print(f"\nAll Strategy Profiles and Utilities:")
    print("=" * 50)
    all_profiles = game.generate_all_strategy_profiles()
    
    print(f"Profile      | u_1    | u_2    | u_3    | u_4    | Total")
    print(f"-------------|--------|--------|--------|--------|--------|--------")
    
    for profile in all_profiles:
        utilities = game.compute_all_utilities(profile)
        total_utility = sum(utilities.values())
        profile_str = f"({profile[1]}, {profile[2]}, {profile[3]}, {profile[4]})"
        
        print(f"{profile_str:12} | {utilities[1]:6.4f} | {utilities[2]:6.4f} | {utilities[3]:6.4f} | {utilities[4]:6.4f}")


if __name__ == "__main__":
    main()
