module LogBin

using Reexport
@reexport using MLBase

using Distributions

export LogisticClassifierBinary
Base.@kwdef mutable struct LogisticClassifierBinary <: ClassifierMixin
    learning_rate::Float64 = 0.05
    max_iter::Int = 200
    λ1::Float64 = 0.0
    λ2::Float64 = 0.0
    w = nothing
    b = nothing
    optimal_w = nothing
    optimal_b = nothing
end

import MLBase: initialiseWeights!
function initialiseWeights!(self::LogisticClassifierBinary, X)
    n_features = size(X,2)
    self.w = rand(Uniform(-1,1), n_features)
    self.b = rand(Uniform(-1,1))
end

import MLBase: forwardPass
forwardPass(self::LogisticClassifierBinary, X) = X*self.w .+ self.b

import MLBase: predict
function predict(self::LogisticClassifierBinary, X)
    # I'm guessing n_targets needs to be 2 for now, until we introduce one_hot stuff.
    z = forwardPass(self, X)
    
    labels = map(x -> (x < 0 ? 1 : 2), z)
end
    

end
