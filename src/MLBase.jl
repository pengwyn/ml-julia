module MLBase

"""Utilities for the neural network modules
"""

using Statistics
using Plots
using ArgCheck
using Data

# Note: all of these functions are defined for single element only as per julia style

export logistic, relu, softmax
# identity is already a function in Base
logistic(x) = 1 / (1 + exp(-x))
# tanh is already a function
relu(x) = x > 0 ? x : zero(x)

# Not sure about this - I think the input is not what is described in the problem.
function softmax(vec)
    out = exp.(vec)
    out /= sum(out)
end

ACTIVATIONS = Dict(:identity => identity,
                   :tanh => tanh,
                   :logistic => logistic,
                   :relu => relu,
                   :softmax => softmax)

			   

# All derivatives can be found with automatic differentiation
using ForwardDiff

DERIVATIVES = Dict(key => x -> ForwardDiff.derivative(func, x)
                   for (key,func) in ACTIVATIONS)


export squaredLoss, logLoss, binaryLogLoss

squaredLoss(y_true, y_pred) = (1 - -y_true*y_pred)^2
# logLoss(y_true, y_pred) = 1/log(2) * log(1 + exp(-y_true*y_pred))
logLoss(y_true, y_pred) = -sum(y_true .* log.(y_pred))
binaryLogLoss(y_true, y_pred) = -(y_true*log(y_pred) + (1-y_true)*log(1 - y_pred))

loss(func, args...) = mean(func.(args...))
# for func in [:squaredLoss, :logLoss, :binaryLogLoss]
for func in [:squaredLoss, :logLoss, :binaryLogLoss]
    @eval $func(y_true::AbstractVector, y_pred::AbstractVector) = loss($func, y_true, y_pred)
end
logLoss(y_true::AbstractMatrix, y_pred::AbstractMatrix) = logLoss(collect(eachrow(y_true)), collect(eachrow(y_pred)))

export ∇y_binaryLogLoss

∇y_binaryLogLoss(y_true::AbstractVector, y_pred::AbstractVector) = @. 1/length(y_true) * (y_pred - y_true) / (y_pred * (1 - y_pred))

# Really these don't belong to anything
correct(y_true, y_pred) = y_true .== y_pred
misclassified(y_true, y_pred) = .!(correct(y_true, y_pred))
score(y_true, y_pred) = mean(correct(y_true, y_pred))


######################################
# * ClassifierMixin
#------------------------------------

export ClassifierMixin
abstract type ClassifierMixin end

macro createVirtualFunc(sym)
    expr = quote
        export $sym
        function $sym end
    end
    esc(expr)
end

@createVirtualFunc initialiseWeights!
@createVirtualFunc forwardPass
@createVirtualFunc predict
@createVirtualFunc fit!
@createVirtualFunc lossFunc

# TODO: Should convert this to a recipe.
export plotFit
using RecipesBase
@recipe function plot(class::ClassifierMixin, cont::DataContainer)
    @argcheck cont.n_features == 2

    X,y = extractArrays(cont)

    return (class,X,y)
end

@recipe function plot(class::ClassifierMixin, X, y_true ; show_lines=false)
    y_pred = forwardPass(class, X)
    pred = predict(class, X)
    tru = oneHotDec(y_true)
    
    matching = correct(tru,pred)

    s = score(pred, tru)

    l = lossFunc(class, X, y_true)
    l = round(l, digits=3)

    xlim = extrema(X[:,1]) |> collect
    xlim .+= 0.1 * [-1,+1] * (xlim[2] - xlim[1])
    ylim = extrema(X[:,2]) |> collect
    ylim .+= 0.1 * [-1,+1] * (ylim[2] - ylim[1])

    xlims --> xlim
    ylims --> ylim
    legend --> false
    title --> "Loss: $l, Score: $s"

    # p = plot()
    
    # if show_lines
    #     # TODO
    #     # Find the line separating these
    #     m = -class.w[1] / class.w[2]
    #     c = -class.b / class.w[2]

    #     func(x) = m*x + c

    #     plot!([func, func],
    #         fillrange=[ylim[1] ylim[2]],
    #         fillcolor=[:green :yellow],
    #         fillalpha=0.2,
    #         color=:black,
    #         linewidth=2)
    # end

    pal = palette(:default)
    pal_light = [RGBA(col.r, col.g, col.b, 0.4) for col in pal]

    markers = ifelse.(matching, :circle, :cross)
    # cols = ifelse.(inds, :red, :blue)
    cols = getindex.(Ref(pal), tru)
    # scatter(X[:,1], X[:,2], color=cols, marker=markers, markerstrokewidth=0, markersize=5)

    xgrid = LinRange(xlim..., 201)
    ygrid = LinRange(ylim..., 201)

    # TODO: fix this
    x_full = [x for x in xgrid for y in ygrid]
    y_full = [y for x in xgrid for y in ygrid]
    X_back = [x_full y_full]
    @show size(X_back)
    tru_back = predict(class, X_back)

    tru_back = reshape(tru_back, length(xgrid), length(ygrid))
    col_back = getindex.(Ref(pal_light), tru_back)

    @series begin
        seriestype := :heatmap
        xgrid, ygrid, col_back
    end

    @series begin
        seriestype := :scatter
        color := cols
        # markerstrokecolor := cols
        # markercolor := nothing
        marker := markers
        markerstrokewidth --> 1
        markersize --> 10

        (X[:,1], X[:,2])
    end
end

################################################
# * Convenience functions
#----------------------------------------------
import Base.dropdims
dropdims(func, args...; dims) = dropdims(func(args..., dims=dims), dims=dims)


end

