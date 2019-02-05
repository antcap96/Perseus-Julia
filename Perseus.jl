module Perseus

using POMDPs
using POMDPPolicies
using POMDPModelTools
using BeliefUpdaters

#TODO: https://github.com/JuliaPOMDP/POMDPToolbox.jl/blob/master/src/model/ordered_spaces.jl
#TODO: https://github.com/JuliaPOMDP/POMDPToolbox.jl/blob/master/src/policies/alpha_vector.jl

export PerseusSolver

struct PerseusSolver <: POMDPs.Solver
    B::Vector{Vector{Float64}}
    improvement_tolerance::Float64
    stop_tolerance::Float64
    max_iteration::Int64
end

function PerseusSolver(pomdp::POMDP; max_iteration::Integer,
                       depth::Integer, branchingfactor::Integer,
                       stop_tolerance=1e-8::AbstractFloat,
                       improvement_tolerance=1e-8::AbstractFloat)

    if branchingfactor != 0
        return PerseusSolver(get_random_beliefs(pomdp, depth, branchingfactor),
                      improvement_tolerance, stop_tolerance, max_iteration)
    else
        return PerseusSolver(get_exhaustive_beliefs(pomdp, depth),
                      improvement_tolerance, stop_tolerance, max_iteration)
    end
end

function POMDPs.solve(solver::PerseusSolver, pomdp::POMDP)
    (αs, policy) = perseus(pomdp, solver.max_iteration,
                           solver.B, stop_tolerance = solver.stop_tolerance,
                           improvement_tolerance = solver.improvement_tolerance)

    return AlphaVectorPolicy(pomdp, collect(αs'), policy)
end

function backup(pomdp::POMDP, b::AbstractVector, α::AbstractMatrix)

    temp = zeros(n_actions(pomdp), n_states(pomdp))

    for a in actions(pomdp)

        for o in observations(pomdp)
            g = zeros(n_states(pomdp))
            current_max = -Inf

            for i in 1:size(α,1)
                temp_g = zeros(n_states(pomdp))

                for (sidx, s) in enumerate(states(pomdp))
                    b[sidx] == 0 && continue
                    sp_dist = transition(pomdp, s, a)

                    for (sp, prob) in weighted_iterator(sp_dist)
                        op_dist = observation(pomdp, a, sp)
                        temp_g[sidx] += pdf(op_dist, o) * prob * α[i, stateindex(pomdp, sp)]
                    end
                end
                val = temp_g' * b #sum along states
                if val > current_max
                    current_max = val
                    current_max_idx = i
                    g = temp_g
                end
            end
            temp[actionindex(pomdp,a),:] .+= g #sum along o
        end

    end

    best_action = :nothing
    best_vector = zeros(n_states(pomdp))
    best_value = -Inf
    for a in actions(pomdp)
        temp2 = Vector{Float64}(undef, n_states(pomdp))
        for s in states(pomdp)
            temp2[stateindex(pomdp,s)] = (reward(pomdp, s, a) 
                                          + discount(pomdp)
                                          * temp[actionindex(pomdp,a),stateindex(pomdp,s)])
        end
        
        value = b' * temp2
        
        if value > best_value
            best_action = a
            best_vector = temp2
            best_value = value
        end
    end

    return (best_vector, best_action)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
end
# V são conjuntos de α's

function perseus_step(pomdp::POMDP{S,A,O}, B::AbstractVector, V_prev::AbstractMatrix, policies_prev::Vector{A}; error=1e-6) where {S,A,O}

    V_ = Vector{Vector{Float64}}()
    policies = Vector{A}()
    B_ = copy(B)

    while size(B_, 1) > 0
        b = rand(B_)

        α, bestaction = backup(pomdp, b, V_prev)

        value_of_b, value_of_b_idx = findmax(V_prev * b)

        if b' * α >= value_of_b
            push!(V_, α)
            push!(policies, bestaction)
        else
            α_ = V_prev[value_of_b_idx, :]
            push!(V_, α_)
            push!(policies, policies_prev[value_of_b_idx])
        end

        filter!(B_) do b
            value_of_b = maximum(V_prev * b)
            for v in V_
                value = v' * b
                (value >= value_of_b || isapprox(value, value_of_b, atol=error)) && return false
            end
            return true
            #V_(b) < value_of_b
        end
    end

    temp = [V_[i][j] for i in 1:size(V_,1), j in 1:size(V_[1],1)]
    #temp = hcat(V_...)' <- TODO this appears to be faster!
    #temp = [v[j] for v in V_, j in 1:size(V_[1],1)] maybe can be done better

    #print(sum([maximum(temp * B[i]) for i in size(B,1)]), "\r")

    return (temp, policies)
end

function perseus_step_all(pomdp::POMDP{S,A,O}, B::AbstractVector, V_prev::AbstractMatrix, policies_prev::Vector{A}; error=1e-6) where {S,A,O}

    V_ = Vector{Vector{Float64}}()
    policies = Vector{A}()
    B_ = copy(B)

    for b in B_
        #b = rand(B_)

        α, bestaction = backup(pomdp, b, V_prev)

        value_of_b, value_of_b_idx = findmax(V_prev * b)

        if (b' * α > value_of_b) && !isapprox(b' * α, value_of_b, atol=error)
            push!(V_, α)
            push!(policies, bestaction)
        end
    end
    if length(policies) > 0
        temp = [V_[i][j] for i in 1:size(V_,1), j in 1:size(V_[1],1)]
    else
        temp = Array{Float64, 2}(undef,0,0)
    end
    #temp = hcat(V_...)' <- TODO this appears to be faster!
    #temp = [v[j] for v in V_, j in 1:size(V_[1],1)] maybe can be done better

    #print(sum([maximum(temp * B[i]) for i in size(B,1)]), "\r")

    return (temp, policies)
end

function get_random_beliefs(pomdp::POMDP, depth::Integer, branchingfactor::Integer; error=1e-8)
    b = DiscreteBelief(pomdp,[pdf(initialstate_distribution(pomdp),s) for s in states(pomdp)])
    B = [b]

    first = firstindex(B)
    last = lastindex(B)

    up = DiscreteUpdater(pomdp)

    for d in 1:depth
        for i in first:last
            for j in 1:branchingfactor
                #push!(B, update(up, B[i], rand(actions(pomdp)), rand(observations(pomdp))))
                new_b = update(up, B[i], rand(actions(pomdp)), rand(observations(pomdp)))
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
        B = unique(B)
        first = last+1
        last = lastindex(B)
    end

    return map(b->b.b, B)
end

function observable(pomdp::POMDP, belief, a, o)
    for (sidx, s) in enumerate(belief.b)
        s > 0 || continue
        sp_dist = transition(pomdp, belief.state_list[sidx], a)
        for (sp, p) in weighted_iterator(sp_dist)
            op_dist = observation(pomdp, sp)
            if pdf(op_dist, o) > 0
                return true
            end
        end
    end
    return false
end

#BUG: assumes all actions are possible at all points
function get_exhaustive_beliefs(pomdp::POMDP, depth::Integer; error=1e-8)
    b = DiscreteBelief(pomdp,[pdf(initialstate_distribution(pomdp),s) for s in states(pomdp)])
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
                 B; improvement_tolerance, stop_tolerance, verbose=false) where {S,A,O}
    #show(Base.stdout, MIME"text/plain"(), B)
    # how good it is
    quality = 1/(1-discount(pomdp)) * minimum([reward(pomdp, s, a) for s=states(pomdp), a=actions(pomdp)]) * size(B,1)

    value = 1/(1-discount(pomdp)) * minimum([reward(pomdp, s, a) for s=states(pomdp), a=actions(pomdp)])
    V₀ = fill(value, (1, n_states(pomdp)))

    if n == 1
        return V₀
    end

    Vi, policies = perseus_step(pomdp, B, V₀, A[], error=improvement_tolerance)

    optimal = false

    for i in 1:(n-1)
        Vi, policies = perseus_step(pomdp, B, Vi, policies, error=improvement_tolerance)

        quality_new = sum([maximum(Vi * B[i]) for i in 1:size(B,1)])

        verbose && println("quality: $quality_new, old $quality")

        verbose && flush(stdout)
        if (quality_new - quality) < stop_tolerance
            #BUG: I'm not sure this is here doing anything
            Vi_extra, policies_extra = perseus_step_all(pomdp, B, Vi, policies, error=improvement_tolerance)
            if length(policies_extra) == 0
                verbose && println("no improvements")
                optimal = true
                break
            else
                verbose && println("new improvements: $(length(policies_extra))")
                #println(Vi_extra)
                #println(policies_extra)
                Vi = vcat(Vi, Vi_extra)
                policies = vcat(policies, policies_extra)
            end
            quality_new = sum([maximum(Vi * B[i]) for i in 1:size(B,1)])
            if (quality_new - quality) < stop_tolerance
                verbose && println("quality no better: $quality_new, old $quality")
                optimal = true
                break
            end
            verbose && println("quality beter: $quality_new, old $quality")
        end

        quality = quality_new
    end

    if !optimal
        @warn("perseus did not find reach optimality")
    end

    return (Vi, policies)
end
end