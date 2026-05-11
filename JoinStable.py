import math
from itertools import product
from typing import Dict, List, Set, Tuple, Any
from HedonicProjectGame import HedonicProjectGame

class JoinStable(HedonicProjectGame):
    """
    Implementation of Join Stability (also called Move-in Stability) for Project Hedonic Games.
    
    A strategy profile s is join stable (JS) if it is Nash Stable (NS), and for all i ∈ N, j ≠ s_i:
    if u_i(s) = u_i(j, s_{-i}), then for all ℓ ∈ C_j(s),  
        u_ℓ(s) ≥ u_ℓ(j, s_{-i})
    
    where C_j(s) denotes the set of players who selected project j in the strategy profile s.
    
    Join stability ensures that:
    1. No unilateral deviation benefits the deviator (Nash stability)
    2. No member of the target coalition would benefit from the deviator joining
       (i.e., the members of the target coalition do not prefer the coalition with the deviator 
       over the original coalition)
    """
    
    def __init__(self, players: List[int], projects: List[Any], rewards: Dict[Any, float], 
                 preferences: Dict[int, Dict[frozenset, float]]):
        """
        Initialize a Join Stable Project Hedonic Game.
        
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
    
    def _is_nash_stable(self, strategy_profile: Dict[int, Any]) -> bool:
        """
        Check if a strategy profile is Nash stable.
        This is a prerequisite for join stability.
        
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
    
    def is_join_stable(self, strategy_profile: Dict[int, Any]) -> bool:
        """
        Check if a strategy profile is join stable according to the definition:
        
        A strategy profile s is join stable if it is NS, and for all i ∈ N, j ≠ s_i:
        if u_i(s) = u_i(j, s_{-i}), then for all ℓ ∈ C_j(s),  
            u_ℓ(s) ≥ u_ℓ(j, s_{-i})
        
        Args:
            strategy_profile: The strategy profile to check
            
        Returns:
            True if the profile is join stable, False otherwise
        """
        # First check Nash stability (prerequisite)
        if not self._is_nash_stable(strategy_profile):
            return False
        
        # For each potential deviating player i
        for deviating_player in self.players:
            current_project_i = strategy_profile[deviating_player]
            
            # Compute current utility for deviating player
            current_utility_i = self.compute_utility(strategy_profile, deviating_player)
            
            # For each project j ≠ s_i that player i could deviate to
            for target_project in self.projects:
                if target_project == current_project_i:
                    continue  # Skip current project (no deviation)
                
                # Create deviation profile (j, s_{-i})
                deviation_profile = self.create_deviation_profile(
                    strategy_profile, deviating_player, target_project
                )
                
                # Compute utility after deviation for player i
                deviation_utility_i = self.compute_utility(deviation_profile, deviating_player)
                
                # Check if u_i(s) = u_i(j, s_{-i})
                if abs(current_utility_i - deviation_utility_i) <= 1e-10:
                    # Player i is indifferent, check join stability condition
                    # Get C_j(s): coalition of players currently in project j
                    current_coalition_j = self.get_coalition_for_project(strategy_profile, target_project)
                    
                    # For all ℓ ∈ C_j(s), check that u_ℓ(s) ≥ u_ℓ(j, s_{-i})
                    for affected_player in current_coalition_j:
                        # Compute ℓ's utility in current profile s
                        current_utility_l = self.compute_utility(strategy_profile, affected_player)
                        
                        # Compute ℓ's utility after player i deviates to j
                        deviation_utility_l = self.compute_utility(deviation_profile, affected_player)
                        
                        # Check join stability condition: u_ℓ(s) ≥ u_ℓ(j, s_{-i})
                        # Violation occurs when u_ℓ(s) < u_ℓ(j, s_{-i})
                        if current_utility_l < deviation_utility_l - 1e-10:
                            # Join stability condition violated: found ℓ that benefits from i joining
                            return False
        
        return True
    
    def find_all_join_stable(self) -> List[Dict[int, Any]]:
        """
        Find all join stable strategy profiles.
        
        Returns:
            List of all join stable profiles
        """
        join_stable_profiles = []
        all_profiles = self.generate_all_strategy_profiles()
        
        for profile in all_profiles:
            if self.is_join_stable(profile):
                join_stable_profiles.append(profile)
        
        return join_stable_profiles
    
    def analyze_join_violations(self, strategy_profile: Dict[int, Any]) -> List[Dict[str, Any]]:
        """
        Analyze all join stability violations in a given strategy profile.
        
        Violations can be:
        1. Nash stability violations (u_i(j, s_{-i}) > u_i(s))
        2. Join constraint violations when player is indifferent (u_i(j, s_{-i}) = u_i(s) 
           but there exists player ℓ ∈ C_j(s) such that u_ℓ(j, s_{-i}) > u_ℓ(s))
        
        Args:
            strategy_profile: The strategy profile to analyze
            
        Returns:
            List of violation details, empty if join stable
        """
        violations = []
        
        # Check Nash stability first
        if not self._is_nash_stable(strategy_profile):
            # Find Nash violations
            for deviating_player in self.players:
                current_project_i = strategy_profile[deviating_player]
                current_utility_i = self.compute_utility(strategy_profile, deviating_player)
                
                for target_project in self.projects:
                    if target_project == current_project_i:
                        continue
                    
                    deviation_profile = self.create_deviation_profile(
                        strategy_profile, deviating_player, target_project
                    )
                    deviation_utility_i = self.compute_utility(deviation_profile, deviating_player)
                    
                    if deviation_utility_i > current_utility_i + 1e-10:
                        violations.append({
                            'type': 'nash_violation',
                            'deviating_player': deviating_player,
                            'target_project': target_project,
                            'current_utility_i': current_utility_i,
                            'deviation_utility_i': deviation_utility_i,
                            'utility_increase_i': deviation_utility_i - current_utility_i,
                            'deviation_profile': deviation_profile
                        })
        
        # Check join stability conditions for indifferent deviations
        for deviating_player in self.players:
            current_project_i = strategy_profile[deviating_player]
            current_utility_i = self.compute_utility(strategy_profile, deviating_player)
            
            for target_project in self.projects:
                if target_project == current_project_i:
                    continue
                
                deviation_profile = self.create_deviation_profile(
                    strategy_profile, deviating_player, target_project
                )
                deviation_utility_i = self.compute_utility(deviation_profile, deviating_player)
                
                # Check if player i is indifferent
                if abs(current_utility_i - deviation_utility_i) <= 1e-10:
                    # Check join stability condition for all ℓ ∈ C_j(s)
                    current_coalition_j = self.get_coalition_for_project(strategy_profile, target_project)
                    
                    for affected_player in current_coalition_j:
                        current_utility_l = self.compute_utility(strategy_profile, affected_player)
                        deviation_utility_l = self.compute_utility(deviation_profile, affected_player)
                        
                        # Check for join constraint violation: u_ℓ(j, s_{-i}) > u_ℓ(s)
                        if deviation_utility_l > current_utility_l + 1e-10:
                            violations.append({
                                'type': 'join_constraint',
                                'deviating_player': deviating_player,
                                'target_project': target_project,
                                'affected_player': affected_player,
                                'deviating_player_utility_equal': True,
                                'current_utility_i': current_utility_i,
                                'deviation_utility_i': deviation_utility_i,
                                'current_utility_affected': current_utility_l,
                                'new_utility_affected': deviation_utility_l,
                                'utility_increase_affected': deviation_utility_l - current_utility_l,
                                'deviation_profile': deviation_profile
                            })
        
        return violations
    
    def print_join_analysis(self, strategy_profile: Dict[int, Any]):
        """
        Print detailed join stability analysis for a strategy profile.
        
        Args:
            strategy_profile: The strategy profile to analyze
        """
        print(f"\nJoin Stability Analysis for profile {strategy_profile}")
        print("=" * 70)
        
        is_nash = self._is_nash_stable(strategy_profile)
        is_join = self.is_join_stable(strategy_profile)
        print(f"Nash Stable: {'YES' if is_nash else 'NO'}")
        print(f"Join Stable: {'YES' if is_join else 'NO'}")
        
        # Show current utilities and coalitions
        current_utilities = self.compute_all_utilities(strategy_profile)
        print(f"\nCurrent utilities and coalitions:")
        for project in self.projects:
            coalition = self.get_coalition_for_project(strategy_profile, project)
            if coalition:
                print(f"  Project {project}: coalition {set(coalition)}")
                for player in coalition:
                    print(f"    Player {player}: utility = {current_utilities[player]:.4f}")
            else:
                print(f"  Project {project}: empty coalition")
        
        # Analyze violations if any
        violations = self.analyze_join_violations(strategy_profile)
        
        if not violations:
            print(f"\nNo join stability violations found.")
            print(f"Definition satisfied: Profile is NS and for all indifferent deviations,")
            print(f"no target coalition member benefits from the deviator joining.")
        else:
            print(f"\nJoin stability violations:")
            print(f"Found {len(violations)} violation(s):")
            
            for i, violation in enumerate(violations, 1):
                print(f"\n  Violation {i}:")
                
                if violation['type'] == 'nash_violation':
                    dev_player = violation['deviating_player']
                    target_proj = violation['target_project']
                    current_util = violation['current_utility_i']
                    new_util = violation['deviation_utility_i']
                    increase = violation['utility_increase_i']
                    
                    print(f"    Type: Nash stability violation")
                    print(f"    Player {dev_player} can improve by moving to project {target_proj}:")
                    print(f"      Current utility: {current_util:.4f}")
                    print(f"      New utility: {new_util:.4f}")
                    print(f"      Improvement: +{increase:.4f}")
                    print(f"      Problem: u_i(j, s_{{-i}}) > u_i(s) violates Nash stability")
                
                elif violation['type'] == 'join_constraint':
                    dev_player = violation['deviating_player']
                    target_proj = violation['target_project']
                    affected_player = violation['affected_player']
                    current_util = violation['current_utility_affected']
                    new_util = violation['new_utility_affected']
                    increase = violation['utility_increase_affected']
                    
                    print(f"    Type: Join constraint violation")
                    print(f"    Player {dev_player} is indifferent about moving to project {target_proj}")
                    print(f"    But Player {affected_player} in target coalition would benefit from i joining:")
                    print(f"      Current utility: {current_util:.4f}")
                    print(f"      New utility: {new_util:.4f}")
                    print(f"      Increase: +{increase:.4f}")
                    print(f"      Problem: Join constraint violated (∃ℓ ∈ C_j(s): u_ℓ(j, s_{{-i}}) > u_ℓ(s))")
                
                # Show the coalition change
                current_coalition = self.get_coalition_for_project(strategy_profile, violation['target_project'])
                new_coalition = self.get_coalition_for_project(violation['deviation_profile'], violation['target_project'])
                print(f"      Coalition change: {set(current_coalition)} → {set(new_coalition)}")
    
    def compare_stability_concepts(self, strategy_profile: Dict[int, Any]) -> Dict[str, bool]:
        """
        Compare different stability concepts for a given profile.
        
        Args:
            strategy_profile: The strategy profile to analyze
            
        Returns:
            Dictionary showing which stability concepts are satisfied
        """
        result = {
            'Nash Stable': self._is_nash_stable(strategy_profile),
            'Join Stable': self.is_join_stable(strategy_profile)
        }
        
        # If other stability concepts are available, include them
        try:
            from ProtectiveStable import ProtectiveStable
            protective_game = ProtectiveStable(self.players, self.projects, self.rewards, self.preferences)
            result['Protective Stable'] = protective_game.is_protective_stable(strategy_profile)
        except ImportError:
            result['Protective Stable'] = None
        
        return result
    
    def find_weakly_join_stable(self) -> List[Dict[int, Any]]:
        """
        Find profiles that satisfy a weaker version of join stability.
        This is useful for analysis when no fully join stable profiles exist.
        
        A profile is weakly join stable if violations are minimal.
        
        Returns:
            List of profiles with minimal join stability violations
        """
        all_profiles = self.generate_all_strategy_profiles()
        profile_violations = []
        
        for profile in all_profiles:
            violations = self.analyze_join_violations(profile)
            profile_violations.append((profile, len(violations)))
        
        # Find minimum number of violations
        min_violations = min(count for _, count in profile_violations)
        
        # Return profiles with minimum violations
        weakly_stable = [profile for profile, count in profile_violations 
                        if count == min_violations]
        
        return weakly_stable
    
    def verify_theoretical_relationship(self) -> Dict[str, Any]:
        """
        Analyze the distribution of join stable profiles and their relationship with other stability concepts.
        
        Returns:
            Dictionary with analysis results
        """
        all_profiles = self.generate_all_strategy_profiles()
        js_profiles = []
        ns_profiles = []
        ps_profiles = []
        js_and_ns = []
        js_and_ps = []
        ns_and_ps = []
        all_three = []
        
        # Check if ProtectiveStable is available
        try:
            from ProtectiveStable import ProtectiveStable
            protective_game = ProtectiveStable(self.players, self.projects, self.rewards, self.preferences)
            ps_available = True
        except ImportError:
            ps_available = False
        
        for profile in all_profiles:
            is_join = self.is_join_stable(profile)
            is_nash = self._is_nash_stable(profile)
            is_protective = protective_game.is_protective_stable(profile) if ps_available else False
            
            if is_join:
                js_profiles.append(profile)
            if is_nash:
                ns_profiles.append(profile)
            if is_protective and ps_available:
                ps_profiles.append(profile)
            
            if is_join and is_nash:
                js_and_ns.append(profile)
            if is_join and is_protective and ps_available:
                js_and_ps.append(profile)
            if is_nash and is_protective and ps_available:
                ns_and_ps.append(profile)
            if is_join and is_nash and is_protective and ps_available:
                all_three.append(profile)
        
        result = {
            'total_profiles': len(all_profiles),
            'join_stable_count': len(js_profiles),
            'nash_stable_count': len(ns_profiles),
            'join_profiles': js_profiles,
            'nash_profiles': ns_profiles,
            'js_and_ns': js_and_ns,
            'js_and_ns_count': len(js_and_ns)
        }
        
        if ps_available:
            result.update({
                'protective_stable_count': len(ps_profiles),
                'protective_profiles': ps_profiles,
                'js_and_ps': js_and_ps,
                'ns_and_ps': ns_and_ps,
                'all_three': all_three,
                'js_and_ps_count': len(js_and_ps),
                'ns_and_ps_count': len(ns_and_ps),
                'all_three_count': len(all_three)
            })
        
        return result


def create_join_example_1():
    """
    Create Example 1 as a Join Stable game using additivity separable preferences.
    """
    # Import here to avoid circular imports
    try:
        from AdditivitySeparableGame import create_additivity_separable_example
        additive_game = create_additivity_separable_example()
        return JoinStable(
            additive_game.players, 
            additive_game.projects, 
            additive_game.rewards, 
            additive_game.preferences
        )
    except ImportError:
        # Fallback to original example if import fails
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
        
        return JoinStable(players, projects, rewards, preferences)


def main():
    """Demonstrate join stability analysis."""
    print("=" * 70)
    print("JOIN STABILITY IN PROJECT HEDONIC GAMES")
    print("=" * 70)
    
    # Create the game
    game = create_join_example_1()
    game.print_game_info()
    
    print(f"\nDefinition: A strategy profile s is join stable if it is NS, and for all i ∈ N, j ≠ s_i:")
    print(f"if u_i(s) = u_i(j, s_{{-i}}), then for all ℓ ∈ C_j(s), u_ℓ(s) ≥ u_ℓ(j, s_{{-i}})")
    print(f"where C_j(s) is the coalition of players in project j in strategy profile s.")
    print(f"\nJoin stability ensures no unilateral deviation benefits the deviator,")
    print(f"and no member of the target coalition would benefit from the deviator joining.")
    
    # Find all join stable profiles
    print(f"\n" + "="*70)
    print("FINDING ALL JOIN STABLE PROFILES")
    print("="*70)
    
    join_profiles = game.find_all_join_stable()
    
    if not join_profiles:
        print("No join stable profiles exist.")
        
        # Find weakly join stable profiles
        print("\nFinding weakly join stable profiles (minimal violations)...")
        weakly_stable = game.find_weakly_join_stable()
        
        if weakly_stable:
            print(f"Found {len(weakly_stable)} weakly join stable profile(s):")
            for i, profile in enumerate(weakly_stable[:3], 1):  # Show first 3
                violations = game.analyze_join_violations(profile)
                print(f"  Profile {i}: {profile} ({len(violations)} violations)")
    else:
        print(f"Found {len(join_profiles)} join stable profile(s):")
        
        for i, profile in enumerate(join_profiles, 1):
            print(f"\nJoin stable profile {i}: {profile}")
            utilities = game.compute_all_utilities(profile)
            for player, utility in utilities.items():
                print(f"  Player {player} utility: {utility:.4f}")
    
    # Analyze all strategy profiles
    print(f"\n" + "="*70)
    print("COMPLETE JOIN STABILITY ANALYSIS")
    print("="*70)
    
    all_profiles = game.generate_all_strategy_profiles()
    print(f"Analyzing all {len(all_profiles)} possible strategy profiles:")
    
    print(f"\nProfile           | Nash    | Join    | u_1    | u_2    | u_3    | u_4    | Violations")
    print(f"------------------|---------|---------|--------|--------|--------|--------|------------")
    
    join_count = 0
    nash_count = 0
    for profile in all_profiles:
        is_nash = game._is_nash_stable(profile)
        is_join = game.is_join_stable(profile)
        violations = game.analyze_join_violations(profile)
        
        if is_nash:
            nash_count += 1
        if is_join:
            join_count += 1
        
        utilities = game.compute_all_utilities(profile)
        profile_str = f"({profile[1]}, {profile[2]}, {profile[3]}, {profile[4]})"
        nash_str = "YES" if is_nash else "NO"
        join_str = "YES" if is_join else "NO"
        
        print(f"{profile_str:17} | {nash_str:7} | {join_str:7} | {utilities[1]:6.4f} | {utilities[2]:6.4f} | {utilities[3]:6.4f} | {utilities[4]:6.4f} | {len(violations):10}")
    
    print(f"\nSummary: {nash_count}/{len(all_profiles)} profiles are Nash stable")
    print(f"Summary: {join_count}/{len(all_profiles)} profiles are join stable")
    
    # Detailed analysis of interesting profiles
    print(f"\n" + "="*70)
    print("DETAILED JOIN STABILITY ANALYSIS")
    print("="*70)
    
    # Analyze the initial profile from the paper
    initial_profile = {1: 'a', 2: 'a', 3: 'a', 4: 'a'}
    game.print_join_analysis(initial_profile)
    
    # Analyze a profile with violations for comparison
    violation_profile = {1: 'a', 2: 'b', 3: 'a', 4: 'b'}  # This should have violations
    game.print_join_analysis(violation_profile)
    
    # Analyze join stability distribution and relationship with other stability concepts
    print(f"\n" + "="*70)
    print("JOIN STABILITY THEORETICAL ANALYSIS")
    print("="*70)
    
    analysis = game.verify_theoretical_relationship()
    print(f"Analyzing relationship between Join Stability and other stability concepts")
    print(f"Total strategy profiles: {analysis['total_profiles']}")
    print(f"Nash stable profiles: {analysis['nash_stable_count']}")
    print(f"Join stable profiles: {analysis['join_stable_count']}")
    print(f"Both NS and JS: {analysis['js_and_ns_count']}")
    print(f"Percentage of profiles that are Nash stable: {analysis['nash_stable_count']/analysis['total_profiles']*100:.1f}%")
    print(f"Percentage of profiles that are join stable: {analysis['join_stable_count']/analysis['total_profiles']*100:.1f}%")
    
    if 'protective_stable_count' in analysis:
        print(f"Protective stable profiles: {analysis['protective_stable_count']}")
        print(f"JS and PS: {analysis['js_and_ps_count']}")
        print(f"NS and PS: {analysis['ns_and_ps_count']}")
        print(f"All three (NS, JS, PS): {analysis['all_three_count']}")
        print(f"Percentage of profiles that are protective stable: {analysis['protective_stable_count']/analysis['total_profiles']*100:.1f}%")
    
    # Compare stability concepts if possible
    print(f"\n" + "="*70)
    print("STABILITY CONCEPT COMPARISON")
    print("="*70)
    
    print(f"Comparing stability concepts for initial profile {initial_profile}:")
    comparison = game.compare_stability_concepts(initial_profile)
    for concept, stable in comparison.items():
        if stable is not None:
            status = "YES" if stable else "NO"
            print(f"  {concept}: {status}")
        else:
            print(f"  {concept}: Not available")


if __name__ == "__main__":
    main()
