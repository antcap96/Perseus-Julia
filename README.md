# Perseus

This package implements the Perseus solver for POMDPs.

Perseus is a point-based value iteration algorithm.

## Usage

```julia
using POMDPs
using Perseus
using POMDPModels
pomdp = TigerPOMDP()

# setting branchingfactor to 0 does an exhaustive search up to depth
solver = PerseusSolver(pomdp, max_iterations=10000, depth=6, branchingfactor=0, verbose=true)

policy = solve(solver, pomdp)
```
