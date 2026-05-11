"""
Best-Response Dynamics for Computing NS/JS/LS Outcomes in Hedonic Project Games

This module implements best-response dynamics algorithms to compute stable outcomes
and measure convergence time, addressing the gap in the paper which doesn't provide
algorithms for computing these stability concepts.

Key Features:
- Multiple player selection strategies (round-robin, random, max-gain)
- Convergence detection (Nash equilibrium, cycles, timeout)
- Metrics: iterations, time, improvement steps, cycles
- Support for NS, JS, and LS computation
"""

import time
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import statistics

from HedonicProjectGame import HedonicProjectGame
from NashStable import NashStable
from JoinStable import JoinStable
from LeaveStable import LeaveStable


class PlayerSelectionRule(Enum):
    """Strategy for selecting which player moves next in best-response dynamics."""
    ROUND_ROBIN = "round_robin"      # Players move in fixed order
    RANDOM = "random"                 # Random player selection
    MAX_GAIN = "max_gain"            # Player with highest potential gain moves
    MIN_REGRET = "min_regret"        # Player with highest regret (current vs best) moves


class ConvergenceStatus(Enum):
    """Result of best-response dynamics execution."""
    CONVERGED = "converged"          # Reached a stable profile
    CYCLE_DETECTED = "cycle"         # Detected a cycle in dynamics
    MAX_ITERATIONS = "max_iter"      # Hit iteration limit
    TIMEOUT = "timeout"              # Hit time limit


@dataclass
class DynamicsResult:
    """Result of running best-response dynamics."""
    final_profile: Dict[int, Any]
    status: ConvergenceStatus
    iterations: int
    elapsed_time: float
    improvements: int                 # Number of successful improvement steps
    is_ns: bool
    is_js: Optional[bool] = None
    is_ls: Optional[bool] = None
    trajectory: List[Dict[int, Any]] = field(default_factory=list)
    cycle_length: Optional[int] = None
    player_moves: Dict[int, int] = field(default_factory=dict)  # Moves per player
    

@dataclass
class ConvergenceStats:
    """Statistics from multiple dynamics runs."""
    total_runs: int
    converged_runs: int
    cycle_runs: int
    timeout_runs: int
    avg_iterations: float
    std_iterations: float
    avg_time: float
    std_time: float
    avg_improvements: float
    ns_count: int
    js_count: int
    ls_count: int
    unique_outcomes: int


class BestResponseDynamics:
    """
    Implements best-response dynamics for computing stable outcomes in HPGs.
    
    Best-response dynamics: In each step, a player who can improve their utility
    by switching projects makes their best response (moves to their best alternative).
    """
    
    def __init__(self, game: HedonicProjectGame):
        """
        Initialize best-response dynamics for a game.
        
        Args:
            game: The hedonic project game instance
        """
        self.game = game
        self.players = game.players
        self.projects = game.projects
        
        # Create stability checkers
        self.ns_checker = NashStable(game.players, game.projects, 
                                      game.rewards, game.preferences)
        self.js_checker = JoinStable(game.players, game.projects,
                                      game.rewards, game.preferences)
        self.ls_checker = LeaveStable(game.players, game.projects,
                                       game.rewards, game.preferences)
    
    def get_best_response(self, profile: Dict[int, Any], player: int) -> Tuple[Any, float]:
        """
        Find the best response for a player given current profile.
        
        Args:
            profile: Current strategy profile
            player: Player to find best response for
            
        Returns:
            Tuple of (best_project, utility_gain)
        """
        current_utility = self.game.compute_utility(profile, player)
        current_project = profile[player]
        
        best_project = current_project
        best_utility = current_utility
        
        for project in self.projects:
            if project == current_project:
                continue
            
            # Compute utility if player switches to this project
            test_profile = profile.copy()
            test_profile[player] = project
            new_utility = self.game.compute_utility(test_profile, player)
            
            if new_utility > best_utility + 1e-10:
                best_utility = new_utility
                best_project = project
        
        gain = best_utility - current_utility
        return best_project, gain
    
    def get_improving_players(self, profile: Dict[int, Any]) -> List[Tuple[int, Any, float]]:
        """
        Find all players who can improve by deviating.
        
        Args:
            profile: Current strategy profile
            
        Returns:
            List of (player, best_project, utility_gain) for all improving players
        """
        improving = []
        for player in self.players:
            best_proj, gain = self.get_best_response(profile, player)
            if gain > 1e-10:
                improving.append((player, best_proj, gain))
        return improving
    
    def select_player(self, improving_players: List[Tuple[int, Any, float]], 
                      rule: PlayerSelectionRule,
                      round_idx: int = 0) -> Optional[Tuple[int, Any, float]]:
        """
        Select which player moves next based on selection rule.
        
        Args:
            improving_players: List of (player, best_project, gain)
            rule: Selection rule to use
            round_idx: Current round index (for round-robin)
            
        Returns:
            Selected (player, project, gain) or None if no improving players
        """
        if not improving_players:
            return None
        
        if rule == PlayerSelectionRule.ROUND_ROBIN:
            # Find first improving player in round-robin order
            player_order = self.players[round_idx % len(self.players):] + \
                          self.players[:round_idx % len(self.players)]
            for p in player_order:
                for imp in improving_players:
                    if imp[0] == p:
                        return imp
            return improving_players[0]
        
        elif rule == PlayerSelectionRule.RANDOM:
            return random.choice(improving_players)
        
        elif rule == PlayerSelectionRule.MAX_GAIN:
            return max(improving_players, key=lambda x: x[2])
        
        elif rule == PlayerSelectionRule.MIN_REGRET:
            # Same as max_gain in this context
            return max(improving_players, key=lambda x: x[2])
        
        return improving_players[0]
    
    def _profile_to_tuple(self, profile: Dict[int, Any]) -> tuple:
        """Convert profile to hashable tuple for cycle detection."""
        return tuple(profile[p] for p in sorted(profile.keys()))
    
    def run(self, 
            initial_profile: Optional[Dict[int, Any]] = None,
            rule: PlayerSelectionRule = PlayerSelectionRule.MAX_GAIN,
            max_iterations: int = 10000,
            timeout: float = 60.0,
            track_trajectory: bool = False,
            check_js: bool = True,
            check_ls: bool = True) -> DynamicsResult:
        """
        Run best-response dynamics until convergence or stopping condition.
        
        Args:
            initial_profile: Starting profile (random if None)
            rule: Player selection rule
            max_iterations: Maximum iterations before stopping
            timeout: Maximum time in seconds
            track_trajectory: Whether to store all visited profiles
            check_js: Whether to check Join Stability at end
            check_ls: Whether to check Leave Stability at end
            
        Returns:
            DynamicsResult with convergence info and final profile
        """
        # Initialize profile
        if initial_profile is None:
            profile = {p: random.choice(self.projects) for p in self.players}
        else:
            profile = initial_profile.copy()
        
        # Tracking
        start_time = time.time()
        visited = set()
        visited.add(self._profile_to_tuple(profile))
        trajectory = [profile.copy()] if track_trajectory else []
        player_moves = {p: 0 for p in self.players}
        iterations = 0
        improvements = 0
        status = ConvergenceStatus.CONVERGED
        cycle_length = None
        
        while iterations < max_iterations:
            # Check timeout
            if time.time() - start_time > timeout:
                status = ConvergenceStatus.TIMEOUT
                break
            
            # Find improving players
            improving = self.get_improving_players(profile)
            
            if not improving:
                # No one can improve - Nash stable!
                status = ConvergenceStatus.CONVERGED
                break
            
            # Select player and make move
            selected = self.select_player(improving, rule, iterations)
            if selected is None:
                status = ConvergenceStatus.CONVERGED
                break
            
            player, new_project, gain = selected
            profile[player] = new_project
            player_moves[player] += 1
            improvements += 1
            iterations += 1
            
            # Check for cycle
            profile_tuple = self._profile_to_tuple(profile)
            if profile_tuple in visited:
                # Found a cycle - find its length
                if track_trajectory:
                    for i, past_profile in enumerate(trajectory):
                        if self._profile_to_tuple(past_profile) == profile_tuple:
                            cycle_length = len(trajectory) - i
                            break
                else:
                    cycle_length = -1  # Unknown length
                status = ConvergenceStatus.CYCLE_DETECTED
                break
            
            visited.add(profile_tuple)
            if track_trajectory:
                trajectory.append(profile.copy())
        
        if iterations >= max_iterations:
            status = ConvergenceStatus.MAX_ITERATIONS
        
        elapsed = time.time() - start_time
        
        # Check stability of final profile
        is_ns = self.ns_checker.is_nash_stable(profile)
        is_js = self.js_checker.is_join_stable(profile) if check_js else None
        is_ls = self.ls_checker.is_leave_stable(profile) if check_ls else None
        
        return DynamicsResult(
            final_profile=profile,
            status=status,
            iterations=iterations,
            elapsed_time=elapsed,
            improvements=improvements,
            is_ns=is_ns,
            is_js=is_js,
            is_ls=is_ls,
            trajectory=trajectory,
            cycle_length=cycle_length,
            player_moves=player_moves
        )
    
    def run_multiple(self,
                     num_runs: int = 100,
                     rule: PlayerSelectionRule = PlayerSelectionRule.RANDOM,
                     max_iterations: int = 10000,
                     timeout: float = 60.0,
                     check_js: bool = True,
                     check_ls: bool = True) -> Tuple[List[DynamicsResult], ConvergenceStats]:
        """
        Run best-response dynamics multiple times with random starting points.
        
        Args:
            num_runs: Number of runs
            rule: Player selection rule
            max_iterations: Max iterations per run
            timeout: Max time per run
            check_js: Whether to check Join Stability
            check_ls: Whether to check Leave Stability
            
        Returns:
            Tuple of (list of results, aggregate statistics)
        """
        results = []
        unique_outcomes = set()
        
        for _ in range(num_runs):
            result = self.run(
                initial_profile=None,
                rule=rule,
                max_iterations=max_iterations,
                timeout=timeout,
                track_trajectory=False,
                check_js=check_js,
                check_ls=check_ls
            )
            results.append(result)
            if result.status == ConvergenceStatus.CONVERGED:
                unique_outcomes.add(self._profile_to_tuple(result.final_profile))
        
        # Compute statistics
        converged = [r for r in results if r.status == ConvergenceStatus.CONVERGED]
        cycles = [r for r in results if r.status == ConvergenceStatus.CYCLE_DETECTED]
        timeouts = [r for r in results if r.status in 
                    (ConvergenceStatus.TIMEOUT, ConvergenceStatus.MAX_ITERATIONS)]
        
        iterations = [r.iterations for r in results]
        times = [r.elapsed_time for r in results]
        improvements = [r.improvements for r in results]
        
        stats = ConvergenceStats(
            total_runs=num_runs,
            converged_runs=len(converged),
            cycle_runs=len(cycles),
            timeout_runs=len(timeouts),
            avg_iterations=statistics.mean(iterations) if iterations else 0,
            std_iterations=statistics.stdev(iterations) if len(iterations) > 1 else 0,
            avg_time=statistics.mean(times) if times else 0,
            std_time=statistics.stdev(times) if len(times) > 1 else 0,
            avg_improvements=statistics.mean(improvements) if improvements else 0,
            ns_count=sum(1 for r in converged if r.is_ns),
            js_count=sum(1 for r in converged if r.is_js),
            ls_count=sum(1 for r in converged if r.is_ls),
            unique_outcomes=len(unique_outcomes)
        )
        
        return results, stats


class JSAwareDynamics(BestResponseDynamics):
    """
    Best-response dynamics that aims to find Join Stable outcomes.
    
    Extends basic dynamics with JS-aware move selection:
    - Avoids moves where the target coalition would benefit (breaking JS)
    - Prioritizes moves that preserve JS properties
    """
    
    def get_js_compatible_moves(self, profile: Dict[int, Any], player: int) -> List[Tuple[Any, float]]:
        """
        Find moves for a player that don't violate JS conditions.
        
        A move is JS-compatible if:
        1. It improves the player's utility (or keeps it same)
        2. No member of the target coalition strictly benefits from player joining
        """
        current_utility = self.game.compute_utility(profile, player)
        current_project = profile[player]
        compatible = []
        
        for project in self.projects:
            if project == current_project:
                continue
            
            test_profile = profile.copy()
            test_profile[player] = project
            new_utility = self.game.compute_utility(test_profile, player)
            
            # Check if this is an improving move
            if new_utility < current_utility - 1e-10:
                continue
            
            # Check JS condition: no target member should benefit
            target_coalition = self.game.get_coalition_for_project(profile, project)
            js_violation = False
            
            for member in target_coalition:
                old_util = self.game.compute_utility(profile, member)
                new_util = self.game.compute_utility(test_profile, member)
                if new_util > old_util + 1e-10:
                    js_violation = True
                    break
            
            if not js_violation:
                gain = new_utility - current_utility
                compatible.append((project, gain))
        
        return compatible
    
    def run_js_aware(self,
                     initial_profile: Optional[Dict[int, Any]] = None,
                     max_iterations: int = 10000,
                     timeout: float = 60.0) -> DynamicsResult:
        """
        Run JS-aware best-response dynamics.
        
        Players only make JS-compatible moves when possible.
        """
        if initial_profile is None:
            profile = {p: random.choice(self.projects) for p in self.players}
        else:
            profile = initial_profile.copy()
        
        start_time = time.time()
        visited = set()
        visited.add(self._profile_to_tuple(profile))
        player_moves = {p: 0 for p in self.players}
        iterations = 0
        improvements = 0
        status = ConvergenceStatus.CONVERGED
        
        while iterations < max_iterations:
            if time.time() - start_time > timeout:
                status = ConvergenceStatus.TIMEOUT
                break
            
            # Find players with JS-compatible improving moves
            candidates = []
            for player in self.players:
                moves = self.get_js_compatible_moves(profile, player)
                improving = [(proj, gain) for proj, gain in moves if gain > 1e-10]
                if improving:
                    best = max(improving, key=lambda x: x[1])
                    candidates.append((player, best[0], best[1]))
            
            if not candidates:
                # Try regular improving moves as fallback
                candidates = self.get_improving_players(profile)
            
            if not candidates:
                status = ConvergenceStatus.CONVERGED
                break
            
            # Select best candidate
            player, new_project, gain = max(candidates, key=lambda x: x[2])
            profile[player] = new_project
            player_moves[player] += 1
            improvements += 1
            iterations += 1
            
            profile_tuple = self._profile_to_tuple(profile)
            if profile_tuple in visited:
                status = ConvergenceStatus.CYCLE_DETECTED
                break
            visited.add(profile_tuple)
        
        if iterations >= max_iterations:
            status = ConvergenceStatus.MAX_ITERATIONS
        
        elapsed = time.time() - start_time
        is_ns = self.ns_checker.is_nash_stable(profile)
        is_js = self.js_checker.is_join_stable(profile)
        is_ls = self.ls_checker.is_leave_stable(profile)
        
        return DynamicsResult(
            final_profile=profile,
            status=status,
            iterations=iterations,
            elapsed_time=elapsed,
            improvements=improvements,
            is_ns=is_ns,
            is_js=is_js,
            is_ls=is_ls,
            player_moves=player_moves
        )


class LSAwareDynamics(BestResponseDynamics):
    """
    Best-response dynamics that aims to find Leave Stable outcomes.
    
    Extends basic dynamics with LS-aware move selection:
    - Avoids moves where the source coalition would benefit (breaking LS)
    """
    
    def get_ls_compatible_moves(self, profile: Dict[int, Any], player: int) -> List[Tuple[Any, float]]:
        """
        Find moves for a player that don't violate LS conditions.
        
        A move is LS-compatible if:
        1. It improves the player's utility (or keeps it same)
        2. No member of the source coalition strictly benefits from player leaving
        """
        current_utility = self.game.compute_utility(profile, player)
        current_project = profile[player]
        compatible = []
        
        source_coalition = self.game.get_coalition_for_project(profile, current_project)
        
        for project in self.projects:
            if project == current_project:
                continue
            
            test_profile = profile.copy()
            test_profile[player] = project
            new_utility = self.game.compute_utility(test_profile, player)
            
            if new_utility < current_utility - 1e-10:
                continue
            
            # Check LS condition: no source member should benefit from player leaving
            ls_violation = False
            for member in source_coalition:
                if member == player:
                    continue
                old_util = self.game.compute_utility(profile, member)
                new_util = self.game.compute_utility(test_profile, member)
                if new_util > old_util + 1e-10:
                    ls_violation = True
                    break
            
            if not ls_violation:
                gain = new_utility - current_utility
                compatible.append((project, gain))
        
        return compatible


def analyze_convergence(game: HedonicProjectGame, 
                        num_runs: int = 100,
                        verbose: bool = True) -> Dict[str, Any]:
    """
    Comprehensive convergence analysis for a game.
    
    Args:
        game: The hedonic project game
        num_runs: Number of dynamics runs per strategy
        verbose: Whether to print results
        
    Returns:
        Dictionary with analysis results
    """
    dynamics = BestResponseDynamics(game)
    js_dynamics = JSAwareDynamics(game)
    
    results = {}
    
    # Test different selection rules
    rules = [
        PlayerSelectionRule.ROUND_ROBIN,
        PlayerSelectionRule.RANDOM,
        PlayerSelectionRule.MAX_GAIN
    ]
    
    if verbose:
        print("=" * 70)
        print("BEST-RESPONSE DYNAMICS CONVERGENCE ANALYSIS")
        print("=" * 70)
        print(f"Game: {len(game.players)} players, {len(game.projects)} projects")
        print(f"Runs per strategy: {num_runs}")
        print()
    
    for rule in rules:
        _, stats = dynamics.run_multiple(
            num_runs=num_runs,
            rule=rule,
            max_iterations=10000,
            timeout=30.0
        )
        results[rule.value] = stats
        
        if verbose:
            print(f"Strategy: {rule.value}")
            print(f"  Convergence rate: {stats.converged_runs}/{stats.total_runs} "
                  f"({100*stats.converged_runs/stats.total_runs:.1f}%)")
            print(f"  Cycle rate: {stats.cycle_runs}/{stats.total_runs} "
                  f"({100*stats.cycle_runs/stats.total_runs:.1f}%)")
            print(f"  Avg iterations: {stats.avg_iterations:.1f} ± {stats.std_iterations:.1f}")
            print(f"  Avg time: {1000*stats.avg_time:.2f}ms ± {1000*stats.std_time:.2f}ms")
            print(f"  NS outcomes: {stats.ns_count}, JS outcomes: {stats.js_count}, "
                  f"LS outcomes: {stats.ls_count}")
            print(f"  Unique outcomes: {stats.unique_outcomes}")
            print()
    
    # Test JS-aware dynamics
    if verbose:
        print("JS-Aware Dynamics:")
    
    js_results = []
    for _ in range(num_runs):
        result = js_dynamics.run_js_aware(max_iterations=10000, timeout=30.0)
        js_results.append(result)
    
    js_converged = sum(1 for r in js_results if r.status == ConvergenceStatus.CONVERGED)
    js_stable = sum(1 for r in js_results if r.is_js)
    
    results['js_aware'] = {
        'converged': js_converged,
        'js_stable': js_stable,
        'avg_iterations': statistics.mean(r.iterations for r in js_results)
    }
    
    if verbose:
        print(f"  Convergence rate: {js_converged}/{num_runs}")
        print(f"  JS outcomes: {js_stable}/{num_runs}")
        print(f"  Avg iterations: {results['js_aware']['avg_iterations']:.1f}")
    
    return results


def demo_dynamics():
    """Demonstration of best-response dynamics."""
    from NashStable import create_nash_example_1
    
    print("=" * 70)
    print("BEST-RESPONSE DYNAMICS DEMONSTRATION")
    print("=" * 70)
    
    # Create example game
    game = create_nash_example_1()
    dynamics = BestResponseDynamics(game)
    
    # Single run with trajectory tracking
    print("\n--- Single Run with Trajectory ---")
    result = dynamics.run(
        initial_profile={1: 'a', 2: 'b', 3: 'c', 4: 'd'},
        rule=PlayerSelectionRule.MAX_GAIN,
        track_trajectory=True
    )
    
    print(f"Initial: {{1: 'a', 2: 'b', 3: 'c', 4: 'd'}}")
    print(f"Final: {result.final_profile}")
    print(f"Status: {result.status.value}")
    print(f"Iterations: {result.iterations}")
    print(f"Time: {1000*result.elapsed_time:.2f}ms")
    print(f"NS: {result.is_ns}, JS: {result.is_js}, LS: {result.is_ls}")
    print(f"Player moves: {result.player_moves}")
    
    if result.trajectory:
        print(f"\nTrajectory ({len(result.trajectory)} states):")
        for i, prof in enumerate(result.trajectory[:5]):  # First 5
            print(f"  {i}: {prof}")
        if len(result.trajectory) > 5:
            print(f"  ... ({len(result.trajectory) - 5} more)")
    
    # Multiple runs analysis
    print("\n--- Multiple Runs Analysis ---")
    _, stats = dynamics.run_multiple(num_runs=50, rule=PlayerSelectionRule.RANDOM)
    
    print(f"Runs: {stats.total_runs}")
    print(f"Converged: {stats.converged_runs} ({100*stats.converged_runs/stats.total_runs:.0f}%)")
    print(f"Cycles: {stats.cycle_runs}")
    print(f"Avg iterations: {stats.avg_iterations:.1f} ± {stats.std_iterations:.1f}")
    print(f"Unique NS outcomes: {stats.unique_outcomes}")
    print(f"NS: {stats.ns_count}, JS: {stats.js_count}, LS: {stats.ls_count}")


def run_scalability_test():
    """Test convergence time scaling with game size."""
    from itertools import combinations
    from math import comb
    import random
    
    print("=" * 70)
    print("SCALABILITY ANALYSIS: CONVERGENCE TIME VS GAME SIZE")
    print("=" * 70)
    
    results = []
    
    for n_players in [3, 4, 5, 6, 7, 8]:
        for n_projects in [2, 3, 4]:
            if n_projects > n_players:
                continue
            
            # Create random game
            players = list(range(1, n_players + 1))
            projects = [chr(ord('a') + i) for i in range(n_projects)]
            rewards = {p: random.uniform(1, 10) for p in projects}
            
            # Generate random normalized preferences
            preferences = {}
            for player in players:
                other_players = [p for p in players if p != player]
                player_prefs = {}
                
                raw_prefs = []
                for size in range(len(other_players) + 1):
                    for subset in combinations(other_players, size):
                        coalition = frozenset([player] + list(subset))
                        raw_prefs.append((coalition, random.random()))
                
                total = sum(v for _, v in raw_prefs)
                for coalition, val in raw_prefs:
                    player_prefs[coalition] = val / total
                
                preferences[player] = player_prefs
            
            game = HedonicProjectGame(players, projects, rewards, preferences)
            dynamics = BestResponseDynamics(game)
            
            # Run dynamics
            start = time.time()
            _, stats = dynamics.run_multiple(num_runs=20, rule=PlayerSelectionRule.MAX_GAIN)
            elapsed = time.time() - start
            
            results.append({
                'n': n_players,
                'm': n_projects,
                'profiles': n_projects ** n_players,
                'avg_iter': stats.avg_iterations,
                'avg_time': stats.avg_time,
                'total_time': elapsed,
                'converged': stats.converged_runs / stats.total_runs
            })
            
            print(f"n={n_players}, m={n_projects}: "
                  f"profiles={n_projects**n_players:>5}, "
                  f"avg_iter={stats.avg_iterations:>6.1f}, "
                  f"avg_time={1000*stats.avg_time:>6.2f}ms, "
                  f"converge={100*stats.converged_runs/stats.total_runs:>5.1f}%")
    
    return results


if __name__ == "__main__":
    demo_dynamics()
    print("\n")
    analyze_convergence_results = run_scalability_test()
