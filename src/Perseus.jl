module Perseus

using POMDPs
using POMDPPolicies: AlphaVectorPolicy
using POMDPModelTools: weighted_iterator, ordered_actions, ordered_observations, ordered_states
using BeliefUpdaters: DiscreteUpdater, DiscreteBelief
using DataStructures: DefaultDict
using POMDPModelTools: SparseCat

#TODO: https://github.com/JuliaPOMDP/POMDPToolbox.jl/blob/master/src/policies/alpha_vector.jl

export PerseusSolver

struct PerseusSolver <: POMDPs.Solver
    B::Vector{Vector{Float64}}
    tolerance::Float64
    max_iterations::Int64
    verbose::Bool
end

function PerseusSolver(pomdp::POMDP; max_iterations::Integer,
                       depth::Integer, branchingfactor::Integer,
                       tolerance::AbstractFloat=1e-8, verbose::Bool=false)

    if branchingfactor != 0
        return PerseusSolver(get_random_beliefs(pomdp, depth, branchingfactor),
                      tolerance, max_iterations, verbose)
    else
        return PerseusSolver(get_exhaustive_beliefs(pomdp, depth),
                      tolerance, max_iterations, verbose)
    end
end

function POMDPs.solve(solver::PerseusSolver, pomdp::POMDP)

    (αs, policy) = perseus(pomdp,
                           solver.max_iterations,
                           solver.B,
                           tolerance = solver.tolerance,
                           verbose = solver.verbose)


    temp = [α[j] for α in αs, j in 1:size(αs[1],1)]
    
    return AlphaVectorPolicy(pomdp, collect(temp'), policy)
end

function backup(pomdp::POMDP, b::AbstractVector, V::Vector{Vector{Float64}})

    temp = zeros(n_actions(pomdp), n_states(pomdp))

    ord_actions = ordered_actions(pomdp)
    ord_states = ordered_states(pomdp)
    ord_observations = ordered_observations(pomdp)

    temp_g = Vector{Float64}(undef, n_states(pomdp))
    g = Vector{Float64}(undef, n_states(pomdp))

    for (aidx,a) in enumerate(ord_actions)

        for o in observations(pomdp)
            g .= 0. 
            current_max = -Inf

            for α in V
                temp_g .= 0. 
                #this is constant, i'm very dumbo
                for (sidx, s) in enumerate(ord_states)
                    sp_dist = transition(pomdp, s, a)

                    for (sp, prob) in weighted_iterator(sp_dist)
                        o_dist = observation(pomdp, a, sp)
                        temp_g[sidx] += pdf(o_dist, o) * prob * α[stateindex(pomdp, sp)]
                    end
                end
                val = temp_g' * b #sum along states
                if val > current_max
                    current_max = val

                    g .= temp_g
                end
            end
            temp[aidx,:] .+= g #sum along observations
        end
    end

    best_action = actions(pomdp)[1]
    best_vector = zeros(n_states(pomdp))
    best_value = -Inf

    for (aidx,a) in enumerate(ord_actions)
        temp2 = Vector{Float64}(undef, n_states(pomdp))
        for (sidx, s) in enumerate(ord_states)
            temp2[sidx] = (reward(pomdp, s, a) 
                        + discount(pomdp)
                        * temp[aidx, sidx])
        end
        
        value = b' * temp2
        
        if value > best_value
            best_action = a
            best_vector = temp2
            best_value = value
        end
    end

    return (best_vector, best_action)
end

function perseus_step(pomdp::POMDP{S,A,O}, B::AbstractVector, V_prev::Vector{Vector{Float64}}, policies_prev::Vector{A}) where {S,A,O}

    V_ = Vector{Vector{Float64}}()
    policies = Vector{A}()
    B_ = copy(B)

    improvement = 0.

    
    ord_actions = ordered_actions(pomdp)
    ord_states = ordered_states(pomdp)
    ord_observations = ordered_observations(pomdp)


    while size(B_, 1) > 0
        b = rand(B_)

        α, bestaction = backup(pomdp, b, V_prev)

        value_of_b, value_of_b_idx = findmax([V_prev[i]' * b for i in 1:size(V_prev, 1)])

        if b' * α >= value_of_b
            push!(V_, α)
            push!(policies, bestaction)
        else
            α_ = V_prev[value_of_b_idx]
            push!(V_, α_)
            push!(policies, policies_prev[value_of_b_idx])
        end

        filter!(B_) do other_b
            
            value_of_other_b = maximum([V_prev[i]' * other_b for i in 1:size(V_prev, 1)])
            for v in V_
                value = v' * other_b
                if (value >= value_of_other_b)
                    improvement += value - value_of_other_b
                    return false
                end
            end
            return true
        end
    end

    return (V_, policies, improvement/length(B))
end

function perseus_step_all(pomdp::POMDP{S,A,O}, B::AbstractVector, V_prev::Vector{Vector{Float64}}, policies_prev::Vector{A}) where {S,A,O}

    V_ = Vector{Vector{Float64}}()
    policies = Vector{A}()
    B_ = copy(B)

    improvement = 0.

    for b in B_
        α, bestaction = backup(pomdp, b, V_prev)

        value_of_b = maximum([V_prev[i]' * b for i in 1:size(V_prev, 1)])

        if (α' * b > value_of_b)
            if (length(V_) == 0 || (α' * b > maximum(v' * b for v in V_)))
                push!(V_, α)
                push!(policies, bestaction)
            end

            improvement += α' * b - value_of_b
        end
    end

    return (V_, policies, improvement/length(B))
end

function get_random_beliefs(pomdp::POMDP, depth::Integer, branchingfactor::Integer; error=1e-8)
    b = DiscreteBelief(pomdp,[pdf(initialstate_distribution(pomdp),s) for s in states(pomdp)]) #TODO: have it not have 0s?
    B = [b]

    first = firstindex(B)
    last = lastindex(B)

    up = DiscreteUpdater(pomdp)

    for d in 1:depth
        for i in first:last
            for j in 1:branchingfactor

                a = rand(actions(pomdp))
                
                new_b = update(up, B[i], a, rand(observables(pomdp, b, a)))
                new = true
                for b in B
                    if sum(abs.(b.b .- new_b.b)) < error #BUG: this dooesn't seem good. is there a better way?
                        new = false
                        break
                    end
                end
                if new
                    push!(B, new_b)
                end
            end
        end
        
        first = last+1
        last = lastindex(B)
    end

    return map(x->x.b, B)
end



#currently only works for obs(a,sp) and obs(sp)
#define for obs(s,a,sp)
function observables(pomdp::POMDP{S,A,O}, belief, a::A) where {S,A,O}
    
    o_dist = DefaultDict{O, Float64}(0.)
    sp_dist = DefaultDict{S, Float64}(0.)

    for (sidx, s) in enumerate(belief.b)

        s > 0 || continue
        s_sp_dist = transition(pomdp, belief.state_list[sidx], a)

        for (sp, p) in weighted_iterator(s_sp_dist)
            sp_dist[sp] += p*s

            #o_dist = observation(pomdp, belief.state_list[sidx], a, sp)
            #if pdf(o_dist, o) > 0
            #    return true
            #end
        end
    end

    for (sp, p) in sp_dist
        sp_o_dist = observation(pomdp, a, sp)
        for (o, po) in weighted_iterator(sp_o_dist)
            o_dist[o] += po*p
        end
    end

    temp = SparseCat(observations(pomdp), [o_dist[o] for o in observations(pomdp)])

    total = sum(temp.probs)

    temp.probs ./= total

    return temp
end

"""
Return true if for a given belief and action `a` it is possible to observe `o`
"""
function observable(pomdp::POMDP{S,A,O}, belief, a::A, o::O) where {S,A,O}
    for (sidx, s) in enumerate(belief.b)
        s > 0 || continue
        sp_dist = transition(pomdp, belief.state_list[sidx], a)
        for (sp, p) in weighted_iterator(sp_dist)
            o_dist = observation(pomdp, belief.state_list[sidx], a, sp)
            if pdf(o_dist, o) > 0
                return true
            end
        end
    end
    return false
end

function get_exhaustive_beliefs(pomdp::POMDP, depth::Integer; error=1e-8)
    
    initialstate = [pdf(initialstate_distribution(pomdp),s) for s in states(pomdp)]
    b = DiscreteBelief(pomdp, initialstate)
    B = [b]

    first = firstindex(B)
    last = lastindex(B)

    up = DiscreteUpdater(pomdp)

    for d in 1:depth
        for i in first:last
            for a in actions(pomdp), o in observations(pomdp)
                if observable(pomdp, B[i], a, o)
                    new_b = update(up, B[i], a, o)
                    new = true
                    for b in B
                        if sum(abs.(b.b .- new_b.b)) < error
                            new = false
                        end
                    end
                    if new
                        push!(B, new_b)
                    end
                end
            end
        end
        B = unique(B)
        first = last+1
        last = lastindex(B)
    end

    return map(b->b.b, B)
end

function perseus(pomdp::POMDP{S,A,O}, n::Integer,
                 B; tolerance, verbose) where {S,A,O}

    minimum_reward = minimum([reward(pomdp, s, a) for s in states(pomdp), 
                                                      a in actions(pomdp)])

    value = 1/(1-discount(pomdp)) * minimum_reward
    quality = value

    V₀ = [fill(value, (n_states(pomdp)))] 

    if n == 1
        return V₀
    end

    Vi, policies = perseus_step(pomdp, B, V₀, A[])

    optimal = false

    for i in 1:(n-1)
        Vi, policies, improvement = perseus_step(pomdp, B, Vi, policies)
        
        verbose && println("quality: $(quality+improvement), old $quality")
        verbose && flush(stdout)

        if improvement < tolerance
            
            verbose && println("low improvement: attempting exhaustive search")
            
            Vi_extra, policies_extra, improvement = perseus_step_all(pomdp, B, Vi, policies)
            
            if length(policies_extra) == 0

                verbose && println("no improvements")
                optimal = true
                break
            else

                verbose && println("new improvements: $(length(policies_extra))")
                Vi = vcat(Vi, Vi_extra)
                policies = vcat(policies, policies_extra)
            end
            
            if improvement < tolerance

                verbose && println("no significant improvement to quality: $(quality+improvement), old $quality")
                optimal = true
                break
            end
            verbose && println("exhaustive search gave improvements to quality: $(quality+improvement), old $quality")
        end

        quality = quality+improvement
    end

    if !optimal
        @warn("Perseus did not converge")
    end

    return (Vi, policies)
end
end
