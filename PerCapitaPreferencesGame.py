import math
from itertools import product, combinations
from typing import Dict, List, Set, Tuple, Any, Callable
from HedonicProjectGame import HedonicProjectGame


class PerCapitaPreferencesGame(HedonicProjectGame):
    """
    Implementation of Hedonic Project Games with Per-Capita Preferences.
    
    This class implements Hedonic Project Games where preferences are characterized
    by their per-capita behavior.
    
    Mathematical Framework:
    ----------------------
    For each player i in N and each coalition C in N_i, the per-capita preference is:
        p^_i(C) = p_i(C) / |C|
    
    Per-Capita Non-Decreasing:
        For any C' subset C: p^_i(C) >= p^_i(C')
        (Larger coalitions have higher or equal per-capita preference)
    
    Per-Capita Non-Increasing:
        For any C' subset C: p^_i(C) <= p^_i(C')
        (Larger coalitions have lower or equal per-capita preference)
    
    Per-Capita Strictly Increasing/Decreasing:
        The above inequalities are strict for all C' subset C.
    
    Since preferences depend only on coalition size, for C' subset C we have |C'| < |C|,
    so the conditions become:
        Non-decreasing: p_i(k)/k >= p_i(k-1)/(k-1) for k > 1
        Non-increasing: p_i(k)/k <= p_i(k-1)/(k-1) for k > 1
    """
    
    def __init__(self, players: List[int], projects: List[Any], rewards: Dict[Any, float],
                 preference_type: str = "per_capita_non_decreasing",
                 preference_function: str = "quadratic"):
        """
        Initialize a Per-Capita Preferences Hedonic Project Game.
        
        Args:
            players: List of player IDs (N)
            projects: List of project IDs (M)
            rewards: Dictionary mapping project to reward value (r function)
            preference_type: Type of per-capita preference behavior:
                - "per_capita_non_decreasing": p^_i(C) >= p^_i(C') for C' subset C
                - "per_capita_non_increasing": p^_i(C) <= p^_i(C') for C' subset C
                - "per_capita_strictly_increasing": p^_i(C) > p^_i(C') for C' subset C
                - "per_capita_strictly_decreasing": p^_i(C) < p^_i(C') for C' subset C
                - "per_capita_constant": p^_i(C) = p^_i(C') for all C, C'
            preference_function: Base function type:
                For non-decreasing/strictly increasing:
                    - "quadratic": g(k) = k² (strictly increasing per-capita)
                    - "cubic": g(k) = k³ (strictly increasing per-capita)
                    - "linear": g(k) = k (constant per-capita)
                    - "power_1.5": g(k) = k^1.5 (strictly increasing per-capita)
                For non-increasing/strictly decreasing:
                    - "constant": g(k) = 1 (strictly decreasing per-capita)
                    - "sqrt": g(k) = sqrtk (strictly decreasing per-capita)
                    - "log": g(k) = log(k+1) (strictly decreasing per-capita)
                    - "linear": g(k) = k (constant per-capita)
        """
        self.preference_type = preference_type
        self.preference_function = preference_function
        
        # Generate per-capita preferences automatically
        preferences = self._generate_per_capita_preferences(players, preference_type, preference_function)
        
        # Initialize the parent class
        super().__init__(players, projects, rewards, preferences)
        
        # Validate per-capita property
        self._validate_per_capita_property()
        
        # Validate normalization constraint
        self._validate_preference_normalization()
    
    def _generate_per_capita_preferences(self, players: List[int], 
                                         preference_type: str,
                                         preference_function: str) -> Dict[int, Dict[frozenset, float]]:
        """
        Generate preferences satisfying the specified per-capita property.
        
        Mathematical Process:
        --------------------
        1. Generate raw preference values g(k) for each coalition size k
        2. Verify that g(k)/k satisfies the desired per-capita property
        3. Compute normalization factor: Z = Sum_{k=1}^n C(n-1, k-1) × g(k)
        4. Set normalized preferences: p_i(k) = g(k) / Z
        5. Verify: Sum_{k=1}^n C(n-1, k-1) × p_i(k) = 1
        
        Args:
            players: List of player IDs
            preference_type: Type of per-capita property
            preference_function: Base function type
            
        Returns:
            Dictionary mapping player to their preference function over coalitions
        """
        preferences = {}
        n = len(players)
        
        # Get the base preference function g(k)
        pref_func = self._get_preference_function(preference_type, preference_function, n)
        
        # Calculate raw preference values for each coalition size
        size_preferences = {}
        for size in range(1, n + 1):
            size_preferences[size] = pref_func(size)
        
        # Normalize to satisfy Sum_{C in N_i} p_i(C) = 1
        total_weighted_pref = 0
        for size in range(1, n + 1):
            num_coalitions = math.comb(n - 1, size - 1)
            total_weighted_pref += num_coalitions * size_preferences[size]
        
        # Apply normalization
        for size in size_preferences:
            size_preferences[size] /= total_weighted_pref
        
        # Store size-based preferences for efficient access
        self._size_preferences = size_preferences
        
        # Compute per-capita preferences for validation and access
        self._per_capita_preferences = {
            size: size_preferences[size] / size for size in size_preferences
        }
        
        # Generate preferences for all players
        for player in players:
            player_prefs = {}
            
            for coalition_size in range(1, n + 1):
                for coalition_members in combinations(players, coalition_size):
                    if player in coalition_members:
                        coalition = frozenset(coalition_members)
                        player_prefs[coalition] = size_preferences[coalition_size]
            
            preferences[player] = player_prefs
        
        return preferences
    
    def _get_preference_function(self, preference_type: str, preference_function: str, 
                                  n: int) -> Callable[[int], float]:
        """
        Get the base preference function g(k) for generating preferences.
        
        The function g(k) determines raw preferences, which are then normalized.
        The per-capita preference is p^(k) = g(k)/(k × Z) where Z is normalization.
        
        For per-capita non-decreasing: g(k)/k should be non-decreasing in k
        For per-capita non-increasing: g(k)/k should be non-increasing in k
        
        Args:
            preference_type: Type of per-capita property
            preference_function: Specific function type
            n: Number of players
            
        Returns:
            Function g: k → preference value
        """
        # Per-capita non-decreasing / strictly increasing functions
        if preference_type in ["per_capita_non_decreasing", "per_capita_strictly_increasing"]:
            if preference_function == "quadratic":
                # g(k) = k², so p^(k) ~ k (strictly increasing)
                return lambda k: k * k
            
            elif preference_function == "cubic":
                # g(k) = k³, so p^(k) ~ k² (strictly increasing)
                return lambda k: k * k * k
            
            elif preference_function == "power_1.5":
                # g(k) = k^1.5, so p^(k) ~ k^0.5 (strictly increasing)
                return lambda k: math.pow(k, 1.5)
            
            elif preference_function == "linear":
                # g(k) = k, so p^(k) ~ 1 (constant - boundary case)
                return lambda k: float(k)
            
            elif preference_function == "exponential":
                # g(k) = k * exp(α*k), so p^(k) ~ exp(α*k) (strictly increasing)
                alpha = 0.3
                return lambda k: k * math.exp(alpha * k)
            
            else:
                raise ValueError(f"Unknown function '{preference_function}' for {preference_type}")
        
        # Per-capita non-increasing / strictly decreasing functions
        elif preference_type in ["per_capita_non_increasing", "per_capita_strictly_decreasing"]:
            if preference_function == "constant":
                # g(k) = 1, so p^(k) ~ 1/k (strictly decreasing)
                return lambda k: 1.0
            
            elif preference_function == "sqrt":
                # g(k) = sqrtk, so p^(k) ~ 1/sqrtk (strictly decreasing)
                return lambda k: math.sqrt(k)
            
            elif preference_function == "log":
                # g(k) = log(k+1), so p^(k) ~ log(k+1)/k (strictly decreasing)
                return lambda k: math.log(k + 1)
            
            elif preference_function == "linear":
                # g(k) = k, so p^(k) ~ 1 (constant - boundary case)
                return lambda k: float(k)
            
            elif preference_function == "harmonic":
                # g(k) = 1/k, so p^(k) ~ 1/k² (strictly decreasing)
                return lambda k: 1.0 / k
            
            else:
                raise ValueError(f"Unknown function '{preference_function}' for {preference_type}")
        
        # Per-capita constant
        elif preference_type == "per_capita_constant":
            # g(k) = k, so p^(k) ~ 1 (constant)
            return lambda k: float(k)
        
        else:
            raise ValueError(f"Unknown preference type: {preference_type}")
    
    def _validate_per_capita_property(self):
        """
        Validate that preferences satisfy the specified per-capita property.
        
        Per-capita preference: p^_i(C) = p_i(C) / |C|
        
        For per-capita non-decreasing:
            p^_i(k) >= p^_i(k-1) for all k > 1
        
        For per-capita non-increasing:
            p^_i(k) <= p^_i(k-1) for all k > 1
        
        Strict versions require strict inequality.
        """
        n = len(self.players)
        
        for player in self.players:
            player_prefs = self.preferences[player]
            
            # Group by size and verify same preference for same size
            size_to_pref = {}
            for coalition, pref_value in player_prefs.items():
                size = len(coalition)
                if size in size_to_pref:
                    if abs(size_to_pref[size] - pref_value) > 1e-10:
                        raise ValueError(
                            f"Player {player}: Coalitions of size {size} have different preferences"
                        )
                else:
                    size_to_pref[size] = pref_value
            
            # Compute per-capita preferences
            per_capita = {size: pref / size for size, pref in size_to_pref.items()}
            
            # Check per-capita property
            sizes = sorted(per_capita.keys())
            
            for i in range(len(sizes) - 1):
                k1, k2 = sizes[i], sizes[i + 1]  # k1 < k2
                pc1, pc2 = per_capita[k1], per_capita[k2]
                
                if self.preference_type == "per_capita_non_decreasing":
                    # p^(k2) >= p^(k1) since k2 > k1 means C containing k2 elements contains C' with k1
                    if pc2 < pc1 - 1e-10:
                        raise ValueError(
                            f"Player {player}: Per-capita non-decreasing violated. "
                            f"p^({k2}) = {pc2:.6f} < p^({k1}) = {pc1:.6f}"
                        )
                
                elif self.preference_type == "per_capita_strictly_increasing":
                    if pc2 <= pc1 + 1e-10:
                        raise ValueError(
                            f"Player {player}: Per-capita strictly increasing violated. "
                            f"p^({k2}) = {pc2:.6f} <= p^({k1}) = {pc1:.6f}"
                        )
                
                elif self.preference_type == "per_capita_non_increasing":
                    # p^(k2) <= p^(k1)
                    if pc2 > pc1 + 1e-10:
                        raise ValueError(
                            f"Player {player}: Per-capita non-increasing violated. "
                            f"p^({k2}) = {pc2:.6f} > p^({k1}) = {pc1:.6f}"
                        )
                
                elif self.preference_type == "per_capita_strictly_decreasing":
                    if pc2 >= pc1 - 1e-10:
                        raise ValueError(
                            f"Player {player}: Per-capita strictly decreasing violated. "
                            f"p^({k2}) = {pc2:.6f} >= p^({k1}) = {pc1:.6f}"
                        )
                
                elif self.preference_type == "per_capita_constant":
                    if abs(pc2 - pc1) > 1e-10:
                        raise ValueError(
                            f"Player {player}: Per-capita constant violated. "
                            f"p^({k2}) = {pc2:.6f} != p^({k1}) = {pc1:.6f}"
                        )
    
    def _validate_preference_normalization(self):
        """
        Validate that preferences satisfy: Sum_{C in N_i} p_i(C) = 1 for each player i.
        """
        for player in self.players:
            player_prefs = self.preferences[player]
            total_preference = sum(player_prefs.values())
            
            if abs(total_preference - 1.0) > 1e-10:
                raise ValueError(
                    f"Normalization constraint violated for player {player}: "
                    f"Sum p_{player}(C) = {total_preference:.10f} != 1.0"
                )
    
    def get_per_capita_preference(self, player: int, coalition: frozenset) -> float:
        """
        Get the per-capita preference p^_i(C) = p_i(C) / |C|.
        
        Args:
            player: Player ID
            coalition: Coalition (must contain player)
            
        Returns:
            Per-capita preference value
        """
        if player not in coalition:
            raise ValueError(f"Player {player} not in coalition {set(coalition)}")
        
        preference = self.preferences[player][coalition]
        return preference / len(coalition)
    
    def get_per_capita_preference_by_size(self, coalition_size: int) -> float:
        """
        Get per-capita preference for coalitions of given size.
        
        Args:
            coalition_size: Size of coalition
            
        Returns:
            Per-capita preference value
        """
        if coalition_size < 1 or coalition_size > len(self.players):
            raise ValueError(f"Invalid coalition size: {coalition_size}")
        
        return self._per_capita_preferences[coalition_size]
    
    def get_preference_by_coalition_size(self, player: int, coalition_size: int) -> float:
        """
        Get preference value for coalitions of given size.
        
        Args:
            player: Player ID
            coalition_size: Size of coalition
            
        Returns:
            Preference value p_i(|C|)
        """
        if player not in self.players:
            raise ValueError(f"Player {player} not in game")
        
        if coalition_size < 1 or coalition_size > len(self.players):
            raise ValueError(f"Invalid coalition size: {coalition_size}")
        
        return self._size_preferences[coalition_size]
    
    def compute_utility_efficient(self, player: int, strategy_profile: Dict[int, Any]) -> float:
        """
        Efficiently compute utility using size-based preferences.
        
        Args:
            player: Player ID
            strategy_profile: Dictionary mapping players to projects
            
        Returns:
            Utility value for the player
        """
        player_project = strategy_profile[player]
        
        coalition_members = [p for p, proj in strategy_profile.items() if proj == player_project]
        coalition_size = len(coalition_members)
        
        preference = self._size_preferences[coalition_size]
        project_reward = self.rewards[player_project]
        
        # u_i(s) = p_i(C) × r(s_i) / |C|
        return preference * (project_reward / coalition_size)
    
    def get_all_strategy_profiles(self) -> List[Dict[int, Any]]:
        """Generate all possible strategy profiles."""
        player_project_choices = []
        for player in self.players:
            player_project_choices.append([(player, project) for project in self.projects])
        
        all_profiles = []
        for profile_tuple in product(*player_project_choices):
            profile = {player: project for player, project in profile_tuple}
            all_profiles.append(profile)
        
        return all_profiles
    
    def compute_social_welfare(self, strategy_profile: Dict[int, Any]) -> float:
        """Compute social welfare for a strategy profile."""
        utilities = self.compute_all_utilities(strategy_profile)
        return sum(utilities.values())
    
    def display_all_strategy_profiles_table(self):
        """Display table of all strategy profiles with utilities and social welfare."""
        print(f"Utility analysis for all profiles:")
        
        all_profiles = self.get_all_strategy_profiles()
        
        # Dynamic header
        header = "Profile      |"
        separator = "-------------|"
        for player in sorted(self.players):
            header += f" u_{player}(.) |"
            separator += "--------|"
        header += " SW(.)"
        separator += "--------"
        print(header)
        print(separator)
        
        for profile in all_profiles:
            utilities = self.compute_all_utilities(profile)
            social_welfare = sum(utilities.values())
            
            profile_desc = "(" + ",".join(str(profile[p]) for p in sorted(self.players)) + ")"
            row = f"{profile_desc:<12} |"
            
            for player in sorted(self.players):
                row += f" {utilities[player]:6.4f} |"
            row += f" {social_welfare:6.4f}"
            
            print(row)
        
        # Summary statistics
        all_sw = [self.compute_social_welfare(p) for p in all_profiles]
        print(f"\nSUMMARY: Max SW = {max(all_sw):.4f}, Min SW = {min(all_sw):.4f}, "
              f"Avg SW = {sum(all_sw)/len(all_sw):.4f}")
    
    def print_preference_structure(self):
        """Print preference structure showing per-capita property."""
        print(f"Per-Capita Preferences ({self.preference_type}, {self.preference_function}):")
        print("=" * 70)
        
        n = len(self.players)
        
        print("\nPreference Structure by Coalition Size:")
        print("-" * 60)
        print(f"{'Size |C|':<10} {'p_i(|C|)':<15} {'p^_i(|C|) = p/|C|':<20} {'#Coalitions':<12}")
        print("-" * 60)
        
        for size in sorted(self._size_preferences.keys()):
            pref = self._size_preferences[size]
            per_cap = self._per_capita_preferences[size]
            num_coalitions = math.comb(n - 1, size - 1)
            print(f"{size:<10} {pref:<15.6f} {per_cap:<20.6f} {num_coalitions:<12}")
        
        # Verify per-capita property
        print(f"\nPer-Capita Property Verification ({self.preference_type}):")
        print("-" * 50)
        
        sizes = sorted(self._per_capita_preferences.keys())
        for i in range(len(sizes) - 1):
            k1, k2 = sizes[i], sizes[i + 1]
            pc1, pc2 = self._per_capita_preferences[k1], self._per_capita_preferences[k2]
            
            if self.preference_type in ["per_capita_non_decreasing", "per_capita_strictly_increasing"]:
                relation = ">=" if pc2 >= pc1 else "<"
                status = "[OK]" if pc2 >= pc1 - 1e-10 else "[FAIL]"
            elif self.preference_type in ["per_capita_non_increasing", "per_capita_strictly_decreasing"]:
                relation = "<=" if pc2 <= pc1 else ">"
                status = "[OK]" if pc2 <= pc1 + 1e-10 else "[FAIL]"
            else:
                relation = "=" if abs(pc2 - pc1) < 1e-10 else "!="
                status = "[OK]" if abs(pc2 - pc1) < 1e-10 else "[FAIL]"
            
            print(f"  p^({k2}) = {pc2:.6f} {relation} p^({k1}) = {pc1:.6f}  {status}")
        
        # Normalization check
        total = sum(math.comb(n - 1, k - 1) * self._size_preferences[k] for k in range(1, n + 1))
        print(f"\nNormalization: Sum C(n-1,k-1) × p(k) = {total:.10f}")
    
    def check_per_capita_property(self, verbose: bool = True) -> bool:
        """
        Check that per-capita preferences satisfy the specified property.
        
        Args:
            verbose: Whether to print detailed results
            
        Returns:
            True if property is satisfied, False otherwise
        """
        try:
            if verbose:
                print(f"Checking per-capita property: {self.preference_type}")
                print("=" * 60)
            
            self._validate_per_capita_property()
            
            if verbose:
                print("[OK] Per-capita property satisfied!")
                self.print_preference_structure()
            
            return True
            
        except ValueError as e:
            if verbose:
                print(f"[FAIL] Per-capita property violated: {e}")
            return False


def create_per_capita_non_decreasing_example(preference_function: str = "quadratic"):
    """
    Create example game with per-capita non-decreasing preferences.
    
    Args:
        preference_function: "quadratic", "cubic", "power_1.5", "linear", "exponential"
    """
    players = [1, 2, 3]
    projects = ['a', 'b']
    rewards = {'a': 10, 'b': 16}
    
    return PerCapitaPreferencesGame(
        players, projects, rewards,
        preference_type="per_capita_non_decreasing",
        preference_function=preference_function
    )


def create_per_capita_non_increasing_example(preference_function: str = "constant"):
    """
    Create example game with per-capita non-increasing preferences.
    
    Args:
        preference_function: "constant", "sqrt", "log", "linear", "harmonic"
    """
    players = [1, 2, 3]
    projects = ['a', 'b']
    rewards = {'a': 10, 'b': 16}
    
    return PerCapitaPreferencesGame(
        players, projects, rewards,
        preference_type="per_capita_non_increasing",
        preference_function=preference_function
    )


def create_per_capita_strictly_increasing_example(preference_function: str = "quadratic"):
    """Create example with per-capita strictly increasing preferences."""
    players = [1, 2, 3]
    projects = ['a', 'b']
    rewards = {'a': 10, 'b': 16}
    
    return PerCapitaPreferencesGame(
        players, projects, rewards,
        preference_type="per_capita_strictly_increasing",
        preference_function=preference_function
    )


def create_per_capita_strictly_decreasing_example(preference_function: str = "constant"):
    """Create example with per-capita strictly decreasing preferences."""
    players = [1, 2, 3]
    projects = ['a', 'b']
    rewards = {'a': 10, 'b': 16}
    
    return PerCapitaPreferencesGame(
        players, projects, rewards,
        preference_type="per_capita_strictly_decreasing",
        preference_function=preference_function
    )


def main():
    """Demonstrate per-capita preferences hedonic project games."""
    print("Per-Capita Preferences Hedonic Project Games")
    print("=" * 70)
    
    # Example 1: Per-capita non-decreasing (quadratic)
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Per-Capita Non-Decreasing Preferences (Quadratic)")
    print("=" * 70)
    
    game1 = create_per_capita_non_decreasing_example("quadratic")
    print(f"Players: {game1.players}")
    print(f"Projects: {game1.projects}")
    print(f"Rewards: {game1.rewards}")
    
    game1.print_preference_structure()
    print()
    game1.display_all_strategy_profiles_table()
    
    # Example 2: Per-capita non-increasing (constant base)
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Per-Capita Non-Increasing Preferences (Constant)")
    print("=" * 70)
    
    game2 = create_per_capita_non_increasing_example("constant")
    game2.print_preference_structure()
    print()
    game2.display_all_strategy_profiles_table()
    
    # Example 3: Per-capita strictly increasing (cubic)
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Per-Capita Strictly Increasing Preferences (Cubic)")
    print("=" * 70)
    
    game3 = create_per_capita_strictly_increasing_example("cubic")
    game3.print_preference_structure()
    print()
    game3.display_all_strategy_profiles_table()
    
    # Example 4: Per-capita strictly decreasing (sqrt)
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Per-Capita Strictly Decreasing Preferences (Sqrt)")
    print("=" * 70)
    
    game4 = create_per_capita_strictly_decreasing_example("sqrt")
    game4.print_preference_structure()
    print()
    game4.display_all_strategy_profiles_table()
    
    # Comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON OF PER-CAPITA PREFERENCE TYPES")
    print("=" * 70)
    
    print("\n{:<35} | {:^15} | {:^15}".format(
        "Preference Type", "p^(1)", "p^(3)"
    ))
    print("-" * 70)
    
    for ptype, pfunc in [
        ("per_capita_non_decreasing", "quadratic"),
        ("per_capita_strictly_increasing", "cubic"),
        ("per_capita_non_increasing", "constant"),
        ("per_capita_strictly_decreasing", "sqrt"),
        ("per_capita_constant", "linear")
    ]:
        try:
            g = PerCapitaPreferencesGame(
                [1, 2, 3], ['a', 'b'], {'a': 10, 'b': 16}, ptype, pfunc
            )
            pc1 = g._per_capita_preferences[1]
            pc3 = g._per_capita_preferences[3]
            print(f"{ptype:<35} | {pc1:^15.6f} | {pc3:^15.6f}")
        except Exception as e:
            print(f"{ptype:<35} | Error: {e}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
