module Helper

using JuMP
import MathOptInterface as MOI
using Gurobi, CSV, DataFrames, JuMP, LinearAlgebra, Distributions, Random, Plots, Parquet, JSON3

"""
    attach_bound_logger!(model::JuMP.Model)

Attach a Gurobi callback that logs:
- time_hist: runtime (s)
- lb_hist:   best bound (UB for max)
- ub_hist:   best incumbent objective (LB for max)

Returns `(time_hist, lb_hist, ub_hist)` which are filled during `optimize!(model)`.
"""
function attach_bound_logger!(model::JuMP.Model)
    # history vectors
    times       = Float64[]
    best_obj    = Float64[]   # incumbent
    best_bound  = Float64[]   # best bound
    t0 = time()

    function cb(cb_data, cb_where::Cint)
        # IMPORTANT: signature must match what Gurobi.jl calls:
        #   (cb_data::Gurobi.CallbackData, cb_where::Cint)
        #
        # Log only at global MIP progress callbacks
        if cb_where != Gurobi.GRB_CB_MIP
            return
        end

        # Query incumbent and bound via C API
        objbst_ref = Ref{Cdouble}()
        objbnd_ref = Ref{Cdouble}()
        Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIP_OBJBST, objbst_ref)
        Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIP_OBJBND, objbnd_ref)

        push!(times, time() - t0)
        push!(best_obj,   objbst_ref[])
        push!(best_bound, objbnd_ref[])
        return
    end

    # Attach solver-specific callback (note: pass the JuMP *model*, not backend)
    MOI.set(model, Gurobi.CallbackFunction(), cb)

    return (time = times, best_obj = best_obj, best_bound = best_bound)
end

function compute_pairwise_scores(
    df::DataFrame,
    emb_col::String,
    rho::AbstractVector{<:Real};
    norm_type::Symbol = :l2,
)
    n = nrow(df)
    @assert length(rho) == n "rho must have length n = nrow(df)"

    # 1) embeddings μ_i (as Vector{Vector{Float64}})
    emb_list_raw = df[!, emb_col]
    emb_list = [Float64.(v) for v in emb_list_raw]

    # dimension d
    d = length(emb_list[1])

    # 2) dual norm
    dual_norm(v) = norm_type === :l2  ? norm(v, 2) :
                   norm_type === :l1  ? maximum(abs.(v)) :
                   norm_type === :linf ? sum(abs.(v)) :
                   error("Unsupported norm_type: $norm_type. Use :l2, :l1, or :linf.")

    # 3) kappa(||·||) = 1 (l2,l1), d (linf)
    kappa =
        norm_type in (:l2, :l1) ? 1.0 :
        norm_type == :linf      ? d * 1.0 :
        error("Unsupported norm_type: $norm_type. Use :l2, :l1, or :linf.")

    # 4) build A matrix
    A = zeros(Float64, n, n)

    for i in 1:n-1
        μi = emb_list[i]
        dual_μi = dual_norm(μi)
        for j in i+1:n
            μj = emb_list[j]
            dual_μj = dual_norm(μj)

            A[i, j] =
                dot(μi, μj) +
                rho[i] * dual_μj +
                rho[j] * dual_μi +
                rho[i] * rho[j] * kappa
        end
    end

    # 5) extract upper triangle i<j as a vector for stats
    scores = Float64[]
    for i in 1:n-1, j in i+1:n
        push!(scores, A[i, j])
    end

    return A, scores
end;

function summarize_pairwise_scores(scores::AbstractVector{<:Real})
    println("Pairwise A[i,j] stats (i<j):")
    println("  n_pairs = ", length(scores))
    println("  min     = ", minimum(scores))
    println("  q25     = ", quantile(scores, 0.25))
    println("  median  = ", quantile(scores, 0.50))
    println("  q75     = ", quantile(scores, 0.75))
    println("  max     = ", maximum(scores))
    println("  mean    = ", mean(scores))
end;

end; # modual helper
