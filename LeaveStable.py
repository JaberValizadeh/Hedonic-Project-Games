import math
from itertools import product
from typing import Dict, List, Set, Tuple, Any
from HedonicProjectGame import HedonicProjectGame

class LeaveStable(HedonicProjectGame):
    """
    Implementation of Leave Stability (also called Move-out Stability) for Project Hedonic Games.
    
    A strategy profile s is leave stable (LS) if it is Nash Stable (NS), and for all i ∈ N, j ∈ M:
    if u_i(s) = u_i(j, s_{-i}), then for all ℓ ∈ C_{s_i}(s),  
        u_ℓ(s) ≥ u_ℓ(j, s_{-i})
    
    where C_{s_i}(s) denotes the set of players who selected the same project as player i 
    before player i deviates.
    
    Leave stability ensures that:
    1. No unilateral deviation benefits the deviator (Nash stability)
    2. No member of the current coalition would benefit from the deviator leaving
       (i.e., members of the current coalition with the deviator do not prefer 
       the coalition without her)
    """
    
    def __init__(self, players: List[int], projects: List[Any], rewards: Dict[Any, float], 
                 preferences: Dict[int, Dict[frozenset, float]]):
        """
        Initialize a Leave Stable Project Hedonic Game.
        
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
    
    def get_current_coalition_members(self, strategy_profile: Dict[int, Any], 
                                    player: int) -> List[int]:
        """
        Get C_{s_i}(s): the set of players who selected the same project as player i
        before player i deviates (including player i themselves).
        
        Args:
            strategy_profile: Current strategy profile s
            player: Player i whose coalition we want to find
            
        Returns:
            List of players in the same coalition as player i
        """
        player_project = strategy_profile[player]
        return self.get_coalition_for_project(strategy_profile, player_project)
    
    def _is_nash_stable(self, strategy_profile: Dict[int, Any]) -> bool:
        """
        Check if a strategy profile is Nash stable.
        This is a prerequisite for leave stability.
        
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
    
    def is_leave_stable(self, strategy_profile: Dict[int, Any]) -> bool:
        """
        Check if a strategy profile is leave stable according to the definition:
        
        A strategy profile s is leave stable if it is NS, and for all i ∈ N, j ∈ M:
        if u_i(s) = u_i(j, s_{-i}), then for all ℓ ∈ C_{s_i}(s),  
            u_ℓ(s) ≥ u_ℓ(j, s_{-i})
        
        Args:
            strategy_profile: The strategy profile to check
            
        Returns:
            True if the profile is leave stable, False otherwise
        """
        # First check Nash stability (prerequisite)
        if not self._is_nash_stable(strategy_profile):
            return False
        
        # For each potential deviating player i
        for deviating_player in self.players:
            current_project_i = strategy_profile[deviating_player]
            
            # Compute current utility for deviating player
            current_utility_i = self.compute_utility(strategy_profile, deviating_player)
            
            # For each project j that player i could deviate to
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
                    # Player i is indifferent, check leave stability condition
                    # Get C_{s_i}(s): coalition of players currently with player i (including i)
                    current_coalition_members = self.get_current_coalition_members(
                        strategy_profile, deviating_player
                    )
                    
                    # For all ℓ ∈ C_{s_i}(s), check that u_ℓ(s) ≥ u_ℓ(j, s_{-i})
                    for coalition_member in current_coalition_members:
                        # Compute ℓ's utility in current profile s
                        current_utility_l = self.compute_utility(strategy_profile, coalition_member)
                        
                        # Compute ℓ's utility after player i deviates to j
                        deviation_utility_l = self.compute_utility(deviation_profile, coalition_member)
                        
                        # Check leave stability condition: u_ℓ(s) ≥ u_ℓ(j, s_{-i})
                        # Violation occurs when u_ℓ(s) < u_ℓ(j, s_{-i})
                        if current_utility_l < deviation_utility_l - 1e-10:
                            # Leave stability condition violated: found ℓ that benefits from i leaving
                            return False
        
        return True
    
    def find_all_leave_stable(self) -> List[Dict[int, Any]]:
        """
        Find all leave stable strategy profiles.
        
        Returns:
            List of all leave stable profiles
        """
        leave_stable_profiles = []
        all_profiles = self.generate_all_strategy_profiles()
        
        for profile in all_profiles:
            if self.is_leave_stable(profile):
                leave_stable_profiles.append(profile)
        
        return leave_stable_profiles
    
    def analyze_leave_violations(self, strategy_profile: Dict[int, Any]) -> List[Dict[str, Any]]:
        """
        Analyze all leave stability violations in a given strategy profile.
        
        Violations can be:
        1. Nash stability violations (u_i(j, s_{-i}) > u_i(s))
        2. Leave constraint violations when player is indifferent (u_i(j, s_{-i}) = u_i(s) 
           but there exists player ℓ ∈ C_{s_i}(s) such that u_ℓ(j, s_{-i}) > u_ℓ(s))
        
        Args:
            strategy_profile: The strategy profile to analyze
            
        Returns:
            List of violation details, empty if leave stable
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
        
        # Check leave stability conditions for indifferent deviations
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
                    # Check leave stability condition for all ℓ ∈ C_{s_i}(s)
                    current_coalition_members = self.get_current_coalition_members(
                        strategy_profile, deviating_player
                    )
                    
                    for coalition_member in current_coalition_members:
                        current_utility_l = self.compute_utility(strategy_profile, coalition_member)
                        deviation_utility_l = self.compute_utility(deviation_profile, coalition_member)
                        
                        # Check for leave constraint violation: u_ℓ(j, s_{-i}) > u_ℓ(s)
                        if deviation_utility_l > current_utility_l + 1e-10:
                            violations.append({
                                'type': 'leave_constraint',
                                'deviating_player': deviating_player,
                                'target_project': target_project,
                                'affected_player': coalition_member,
                                'deviating_player_utility_equal': True,
                                'current_utility_i': current_utility_i,
                                'deviation_utility_i': deviation_utility_i,
                                'current_utility_affected': current_utility_l,
                                'new_utility_affected': deviation_utility_l,
                                'utility_increase_affected': deviation_utility_l - current_utility_l,
                                'deviation_profile': deviation_profile
                            })
        
        return violations
    
    def print_leave_analysis(self, strategy_profile: Dict[int, Any]):
        """
        Print detailed leave stability analysis for a strategy profile.
        
        Args:
            strategy_profile: The strategy profile to analyze
        """
        print(f"\nLeave Stability Analysis for profile {strategy_profile}")
        print("=" * 70)
        
        is_nash = self._is_nash_stable(strategy_profile)
        is_leave = self.is_leave_stable(strategy_profile)
        print(f"Nash Stable: {'YES' if is_nash else 'NO'}")
        print(f"Leave Stable: {'YES' if is_leave else 'NO'}")
        
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
        violations = self.analyze_leave_violations(strategy_profile)
        
        if not violations:
            print(f"\nNo leave stability violations found.")
            print(f"Definition satisfied: Profile is NS and for all indifferent deviations,")
            print(f"no current coalition member benefits from the deviator leaving.")
        else:
            print(f"\nLeave stability violations:")
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
                
                elif violation['type'] == 'leave_constraint':
                    dev_player = violation['deviating_player']
                    target_proj = violation['target_project']
                    affected_player = violation['affected_player']
                    current_util = violation['current_utility_affected']
                    new_util = violation['new_utility_affected']
                    increase = violation['utility_increase_affected']
                    
                    print(f"    Type: Leave constraint violation")
                    print(f"    Player {dev_player} is indifferent about moving to project {target_proj}")
                    print(f"    But Player {affected_player} in current coalition would benefit from i leaving:")
                    print(f"      Current utility: {current_util:.4f}")
                    print(f"      New utility: {new_util:.4f}")
                    print(f"      Increase: +{increase:.4f}")
                    print(f"      Problem: Leave constraint violated (∃ℓ ∈ C_{{s_i}}(s): u_ℓ(j, s_{{-i}}) > u_ℓ(s))")
                
                # Show the coalition change
                if violation['type'] == 'leave_constraint':
                    current_coalition = self.get_current_coalition_members(strategy_profile, violation['deviating_player'])
                    new_coalition = self.get_current_coalition_members(violation['deviation_profile'], violation['affected_player'])
                    print(f"      Current coalition: {set(current_coalition)}")
                    print(f"      Coalition after {violation['deviating_player']} leaves: {set(new_coalition)}")
                else:
                    # For Nash violations, show the coalition the player is moving to/from
                    current_coalition = self.get_current_coalition_members(strategy_profile, violation['deviating_player'])
                    target_coalition = self.get_coalition_for_project(strategy_profile, violation['target_project'])
                    print(f"      Current coalition: {set(current_coalition)}")
                    print(f"      Target coalition: {set(target_coalition)}")
    
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
            'Leave Stable': self.is_leave_stable(strategy_profile)
        }
        
        # If other stability concepts are available, include them
        try:
            from JoinStable import JoinStable
            join_game = JoinStable(self.players, self.projects, self.rewards, self.preferences)
            result['Join Stable'] = join_game.is_join_stable(strategy_profile)
        except ImportError:
            result['Join Stable'] = None
            
        try:
            from ProtectiveStable import ProtectiveStable
            protective_game = ProtectiveStable(self.players, self.projects, self.rewards, self.preferences)
            result['Protective Stable'] = protective_game.is_protective_stable(strategy_profile)
        except ImportError:
            result['Protective Stable'] = None
        
        return result
    
    def find_weakly_leave_stable(self) -> List[Dict[int, Any]]:
        """
        Find profiles that satisfy a weaker version of leave stability.
        This is useful for analysis when no fully leave stable profiles exist.
        
        A profile is weakly leave stable if violations are minimal.
        
        Returns:
            List of profiles with minimal leave stability violations
        """
        all_profiles = self.generate_all_strategy_profiles()
        profile_violations = []
        
        for profile in all_profiles:
            violations = self.analyze_leave_violations(profile)
            profile_violations.append((profile, len(violations)))
        
        # Find minimum number of violations
        min_violations = min(count for _, count in profile_violations)
        
        # Return profiles with minimum violations
        weakly_stable = [profile for profile, count in profile_violations 
                        if count == min_violations]
        
        return weakly_stable
    
    def verify_theoretical_relationship(self) -> Dict[str, Any]:
        """
        Analyze the distribution of leave stable profiles and their relationship with other stability concepts.
        
        Returns:
            Dictionary with analysis results
        """
        all_profiles = self.generate_all_strategy_profiles()
        ls_profiles = []
        ns_profiles = []
        js_profiles = []
        ps_profiles = []
        
        # Check if other stability concepts are available
        try:
            from JoinStable import JoinStable
            join_game = JoinStable(self.players, self.projects, self.rewards, self.preferences)
            js_available = True
        except ImportError:
            js_available = False
            
        try:
            from ProtectiveStable import ProtectiveStable
            protective_game = ProtectiveStable(self.players, self.projects, self.rewards, self.preferences)
            ps_available = True
        except ImportError:
            ps_available = False
        
        for profile in all_profiles:
            is_leave = self.is_leave_stable(profile)
            is_nash = self._is_nash_stable(profile)
            is_join = join_game.is_join_stable(profile) if js_available else False
            is_protective = protective_game.is_protective_stable(profile) if ps_available else False
            
            if is_leave:
                ls_profiles.append(profile)
            if is_nash:
                ns_profiles.append(profile)
            if is_join and js_available:
                js_profiles.append(profile)
            if is_protective and ps_available:
                ps_profiles.append(profile)
        
        result = {
            'total_profiles': len(all_profiles),
            'leave_stable_count': len(ls_profiles),
            'nash_stable_count': len(ns_profiles),
            'leave_profiles': ls_profiles,
            'nash_profiles': ns_profiles
        }
        
        if js_available:
            result.update({
                'join_stable_count': len(js_profiles),
                'join_profiles': js_profiles
            })
            
        if ps_available:
            result.update({
                'protective_stable_count': len(ps_profiles),
                'protective_profiles': ps_profiles
            })
        
        return result


def create_leave_example_1():
    """
    Create Example 1 as a Leave Stable game using additivity separable preferences.
    """
    # Import here to avoid circular imports
    try:
        from AdditivitySeparableGame import create_additivity_separable_example
        additive_game = create_additivity_separable_example()
        return LeaveStable(
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
        
        return LeaveStable(players, projects, rewards, preferences)


def main():
    """Demonstrate leave stability analysis."""
    print("=" * 70)
    print("LEAVE STABILITY IN PROJECT HEDONIC GAMES")
    print("=" * 70)
    
    # Create the game
    game = create_leave_example_1()
    game.print_game_info()
    
    print(f"\nDefinition: A strategy profile s is leave stable if it is NS, and for all i ∈ N, j ∈ M:")
    print(f"if u_i(s) = u_i(j, s_{{-i}}), then for all ℓ ∈ C_{{s_i}}(s), u_ℓ(s) ≥ u_ℓ(j, s_{{-i}})")
    print(f"where C_{{s_i}}(s) is the coalition of players in the same project as player i.")
    print(f"\nLeave stability ensures no unilateral deviation benefits the deviator,")
    print(f"and no member of the current coalition would benefit from the deviator leaving.")
    
    # Find all leave stable profiles
    print(f"\n" + "="*70)
    print("FINDING ALL LEAVE STABLE PROFILES")
    print("="*70)
    
    leave_profiles = game.find_all_leave_stable()
    
    if not leave_profiles:
        print("No leave stable profiles exist.")
        
        # Find weakly leave stable profiles
        print("\nFinding weakly leave stable profiles (minimal violations)...")
        weakly_stable = game.find_weakly_leave_stable()
        
        if weakly_stable:
            print(f"Found {len(weakly_stable)} weakly leave stable profile(s):")
            for i, profile in enumerate(weakly_stable[:3], 1):  # Show first 3
                violations = game.analyze_leave_violations(profile)
                print(f"  Profile {i}: {profile} ({len(violations)} violations)")
    else:
        print(f"Found {len(leave_profiles)} leave stable profile(s):")
        
        for i, profile in enumerate(leave_profiles, 1):
            print(f"\nLeave stable profile {i}: {profile}")
            utilities = game.compute_all_utilities(profile)
            for player, utility in utilities.items():
                print(f"  Player {player} utility: {utility:.4f}")
    
    # Analyze all strategy profiles
    print(f"\n" + "="*70)
    print("COMPLETE LEAVE STABILITY ANALYSIS")
    print("="*70)
    
    all_profiles = game.generate_all_strategy_profiles()
    print(f"Analyzing all {len(all_profiles)} possible strategy profiles:")
    
    print(f"\nProfile           | Nash    | Leave   | u_1    | u_2    | u_3    | u_4    | Violations")
    print(f"------------------|---------|---------|--------|--------|--------|--------|------------")
    
    leave_count = 0
    nash_count = 0
    for profile in all_profiles:
        is_nash = game._is_nash_stable(profile)
        is_leave = game.is_leave_stable(profile)
        violations = game.analyze_leave_violations(profile)
        
        if is_nash:
            nash_count += 1
        if is_leave:
            leave_count += 1
        
        utilities = game.compute_all_utilities(profile)
        profile_str = f"({profile[1]}, {profile[2]}, {profile[3]}, {profile[4]})"
        nash_str = "YES" if is_nash else "NO"
        leave_str = "YES" if is_leave else "NO"
        
        print(f"{profile_str:17} | {nash_str:7} | {leave_str:7} | {utilities[1]:6.4f} | {utilities[2]:6.4f} | {utilities[3]:6.4f} | {utilities[4]:6.4f} | {len(violations):10}")
    
    print(f"\nSummary: {nash_count}/{len(all_profiles)} profiles are Nash stable")
    print(f"Summary: {leave_count}/{len(all_profiles)} profiles are leave stable")
    
    # Detailed analysis of interesting profiles
    print(f"\n" + "="*70)
    print("DETAILED LEAVE STABILITY ANALYSIS")
    print("="*70)
    
    # Analyze the initial profile from the paper
    initial_profile = {1: 'a', 2: 'a', 3: 'a', 4: 'a'}
    game.print_leave_analysis(initial_profile)
    
    # Analyze a profile with violations for comparison
    violation_profile = {1: 'a', 2: 'b', 3: 'a', 4: 'b'}  # This should have violations
    game.print_leave_analysis(violation_profile)
    
    # Analyze leave stability distribution and relationship with other stability concepts
    print(f"\n" + "="*70)
    print("LEAVE STABILITY THEORETICAL ANALYSIS")
    print("="*70)
    
    analysis = game.verify_theoretical_relationship()
    print(f"Analyzing relationship between Leave Stability and other stability concepts")
    print(f"Total strategy profiles: {analysis['total_profiles']}")
    print(f"Nash stable profiles: {analysis['nash_stable_count']}")
    print(f"Leave stable profiles: {analysis['leave_stable_count']}")
    print(f"Percentage of profiles that are Nash stable: {analysis['nash_stable_count']/analysis['total_profiles']*100:.1f}%")
    print(f"Percentage of profiles that are leave stable: {analysis['leave_stable_count']/analysis['total_profiles']*100:.1f}%")
    
    if 'join_stable_count' in analysis:
        print(f"Join stable profiles: {analysis['join_stable_count']}")
        print(f"Percentage of profiles that are join stable: {analysis['join_stable_count']/analysis['total_profiles']*100:.1f}%")
    
    if 'protective_stable_count' in analysis:
        print(f"Protective stable profiles: {analysis['protective_stable_count']}")
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
