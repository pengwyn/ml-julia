module MLBase

"""Utilities for the neural network modules
"""

using Statistics
using Plots
using ArgCheck

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
logLoss(y_true, y_pred) = 1/log(2) * log(1 + exp(-y_true*y_pred))
binaryLogLoss(y_true, y_pred) = -(y_true*log(y_pred) + (1-y_true)*log(1 - y_pred))

loss(func, args...) = mean(func.(args...))
for func in [:squaredLoss, :logLoss, :binaryLogLoss]
    @eval $func(y_true::AbstractVector, y_pred::AbstractVector) = loss($func, y_true, y_pred)
end

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
function plotFit(self::ClassifierMixin, X, y_true ; full_grid=true)
    n_features = size(X, 2)
    # @argcheck n_features == 2

    y_pred = predict(self, X)

    s = score(y_pred, y_true)

    l = lossFunc(self, X, y_true)
    l = round(l, digits=3)

    # xlim = extrema(X[:,1]) .* 1.1
    # ylim = extrema(X[:,2]) .* 1.1
    xlim = extrema(X[:,1]) |> collect
    xlim .+= 0.1 * [-1,+1] * (xlim[2] - xlim[1])
    ylim = extrema(X[:,2]) |> collect
    ylim .+= 0.1 * [-1,+1] * (ylim[2] - ylim[1])

    p = plot(xlims=xlim,
             ylims=ylim,
             legend=false,
             title="Loss: $l, Score: $s")

    # Find the line separating these
    m = -self.w[1] / self.w[2]
    c = -self.b / self.w[2]

    func(x) = m*x + c

    plot!([func, func],
          fillrange=[ylim[1] ylim[2]],
          fillcolor=[:green :yellow],
          fillalpha=0.2,
          color=:black,
          linewidth=2)

    inds = misclassified(y_true,y_pred)
    cols = ifelse.(inds, :red, :blue)
    markers = ifelse.(y_true .== 0, :circle, :star)
    scatter!(X[:,1], X[:,2], color=cols, marker=markers, markerstrokewidth=0, markersize=5)
end

################################################
# * Convenience functions
#----------------------------------------------
import Base.dropdims
dropdims(func, args...; dims) = dropdims(func(args..., dims=dims), dims=dims)


end

