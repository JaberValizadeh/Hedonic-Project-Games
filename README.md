# Hedonic Project Games

Official repository for the paper:

**“Stability and Efficiency in Hedonic Project Games”**  
by Jaber Valizadeh, Dongmo Zhang, and Omar Mubin  
School of Computing, Data and Mathematical Sciences  
Western Sydney University, Australia

---

## Overview

This repository provides implementations, experiments, and analysis tools for **Hedonic Project Games (HPGs)** — a new coalition formation framework that integrates:

- **subjective coalition preferences** (who agents collaborate with), and
- **divisible project rewards** (what agents work on).

Unlike classical hedonic games or project games, HPGs model environments where agents must jointly reason about:

1. **which project to join**, and  
2. **which coalition to join**.

The framework captures the fundamental trade-off between:

- collaboration benefits,
- coalition compatibility,
- and reward dilution.

Applications include:

- multi-agent collaboration,
- research team formation,
- crowdsourcing systems,
- distributed AI systems,
- task allocation,
- collaborative robotics,
- and decentralized coordination.

---

## Motivation

Existing models typically capture only one side of collaborative decision-making:

| Model | Coalition Preferences | Project Rewards |
|---|---|---|
| Hedonic Games | ✓ | ✗ |
| Project Games | ✗ | ✓ |
| GASP / AS-GGASP | Partial | Limited |
| Hedonic Project Games | ✓ | ✓ |

HPGs unify these dimensions in a single framework.

---

## Hedonic Project Games

A Hedonic Project Game is defined by:

- a set of agents \(N\),
- a set of projects \(M\),
- project rewards \(r(j)\),
- and agent preference functions over coalitions.

Each agent selects exactly one project.

For a strategy profile \(s\):

- the coalition for project \(j\) is:
  
\[
C_j(s) = \{ i \in N : s_i = j \}
\]

- project rewards are shared equally among coalition members,
- and each agent’s utility is:

\[
u_i(s) =
p_i(C_{s_i}(s))
\cdot
\frac{r(s_i)}{|C_{s_i}(s)|}
\]

where:

- \(p_i(C)\) is agent \(i\)’s preference for coalition \(C\),
- \(r(s_i)\) is the reward of the chosen project,
- and \(|C_{s_i}(s)|\) is coalition size.

This multiplicative utility structure creates trade-offs absent from additive models.

---

## Main Contributions

This repository accompanies the IJCAI paper and includes implementations of:

### Stability Concepts

- Nash Stability (NS)
- Joining Stability (JS)
- Leaving Stability (LS)

### Theoretical Results

The paper establishes:

- Non-existence of stable outcomes under unrestricted preferences
- Existence guarantees under structured preferences
- Efficiency analysis via:
  - Price of Anarchy (PoA)
  - Price of Stability (PoS)

### Key Findings

#### Monotonic Decreasing Preferences

- Nash stable outcomes always exist
- \(NS = JS\)
- Leaving stability may fail
- Under identical rewards and concave preferences:

\[
PoS_{NS} = PoS_{JS} = 1
\]

#### Per-Capita Non-Decreasing Preferences

- Socially optimal allocations are Nash stable
- \(NS = LS\)
- Optimal welfare is always achievable

#### Additively Separable Preferences

- Stable outcomes exist
- Price of Anarchy can still be unbounded

---

## Repository Structure

```text
.
├── algorithms/          # Coalition formation algorithms
├── experiments/         # Experimental evaluation scripts
├── games/               # Hedonic project game models
├── stability/           # Stability verification methods
├── learning/            # Online learning and adaptive dynamics
├── datasets/            # Synthetic and real-world datasets
├── figures/             # Generated figures and plots
├── utils/               # Utility/helper functions
├── notebooks/           # Experimental notebooks
├── paper/               # Paper and supplementary material
└── README.md
