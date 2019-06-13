module Data


using Distributions, Random

using ArgCheck
using DataFrames 

####################################
# * Data generators
#----------------------------------

randomRadius(radius, n_samples, noise) = rand(Normal(radius, noise), n_samples)

function distributeSamples(n_samples, n_targets)
    target_samples = fill(n_samples ÷ n_targets, n_targets)

    N_leftover = n_samples % n_targets
    target_samples[1:N_leftover] .+= 1

    y = map(eachindex(target_samples)) do ind
        # Force base 0 numbering for convenience later
        id = ind-1
        fill(id, target_samples[ind])
    end

    y = vcat(y...)

    return target_samples, y
end

export makeDonut
makeDonut(n_targets::Int=2 ; kwds...) = makeDonut(1:n_targets ; kwds...)
function makeDonut(radii::AbstractVector; n_samples=100, noise=0.)
    n_targets = length(radii)

    target_samples,y = distributeSamples(n_samples, n_targets)

    X = Matrix{Int}(undef,0,2)
    for (sample_size,r) in zip(target_samples, radii)
        θ = rand(Uniform(0,2π), sample_size)
        X0 = randomRadius(r, sample_size, noise) .* cos.(θ)
        X1 = randomRadius(r, sample_size, noise) .* sin.(θ)

        X = [X ; [X0 X1]]
    end

    return X,y
end

export makeCloud
function makeCloud(n_targets=2 ; n_features=2, kwds...)
    centres = map(1:n_targets) do i
        rand(Uniform(-1.0, 1.0), n_features)
    end
    radii = rand(Uniform(0, 0.2), n_targets)

    makeCloud(centres, radii ; n_features=n_features, kwds...)
end

function makeCloud(centres, radii ; n_samples=100, n_features=2)
    @argcheck length(centres) == length(radii)
    n_targets = length(centres)

    target_samples,y = distributeSamples(n_samples, n_targets)

    X = map(y) do label
        ind = label+1
        point = rand.(Normal.(centres[ind], radii[ind]))
    end
    X = hcat(X...) |> permutedims

    return X,y
end
        
        
export makeSpiral
makeSpiral(n_targets::Int=2 ; kwds...) = makeSpiral(LinRange(0, 2π, n_targets+1)[1:end-1] ; kwds...)
function makeSpiral(phases::AbstractVector ; n_samples=100, noise=0.05, inner_radius=0, outer_radius=2)
    # n_features is fixed to 2 here.
    n_targets = length(phases)
    target_samples,y = distributeSamples(n_samples, n_targets)


    X = map(target_samples,phases) do target_n,phase
        # Parameteric with variable p
        p = rand(target_n)

        r = @. inner_radius + p * (outer_radius - inner_radius)
        θ = @. phase + 1.75 * p * 2π

        X1 = @. r * sin(θ)
        X2 = @. r * cos(θ)

        X1 += rand(Normal(0, noise), target_n)
        X2 += rand(Normal(0, noise), target_n)

        [X1 X2]
    end

    X = vcat(X...)

    return X,y
end

export makeXor
function makeXor(n_features=2 ; n_samples=100)
    # n_targets is fixed to 2 here.

    # This would be a lot easier if we could just generate points and classify
    # rather than setting the number of samples per target a priori.

    # Yes I'm going to do that instead...

    XorClassify(point) = xor((point .> 0)...) ? 0 : 1

    X = rand(Uniform(-1, 1), n_samples, n_features)

    y = mapslices(XorClassify, X, dims=2)
    y = vec(y)
        
    return X,y
end

export makeMoons
function makeMoons(; n_samples=100, noise=0.05)
    # n_features and n_targets are fixed to 2 here.
    target_samples,y = distributeSamples(n_samples, 2)

    funcs = [θ -> 2*[cos(θ), sin(θ)] - [1,1],
             θ -> 2*[cos(θ), -sin(θ)] + [1,1]]
    X = map(target_samples,funcs) do target_n,func
        θ = rand(Uniform(0,π), target_n)
        hcat(func.(θ)...)
    end
    X = hcat(X...) |> permutedims

    return X,y
end


################################
# * DataContainer
#------------------------------

export DataContainer
Base.@kwdef mutable struct DataContainer
    data_df::DataFrame

    n_samples::Int

    n_features::Int
    feature_names::Vector{Symbol}

    n_targets::Int
    target_names::Vector{Symbol}

    shuffled::Bool = false

    scales = [ones(n_features), zeros(n_features)]
end

function DataContainer(data_features, data_targets ; shuffle_data=true, conv_one_hot=true)
    @argcheck size(data_features, 1) == size(data_targets, 1)

    if conv_one_hot
        @assert ndims(data_targets) == 1
        if length(unique(data_targets)) > 1
            data_targets = oneHotEnc(data_targets)
        end
    end

    # Force matrix
    if ndims(data_targets) == 1
        data_targets = reshape(data_targets, :, 1)
    end

    n_samples,n_features = size(data_features)
    n_samples,n_targets = size(data_targets)

    feature_columns = map(1:n_features) do col
        Symbol("X$col") => data_features[:,col]
    end

    target_columns = map(1:n_targets) do col
        Symbol("y$col") => data_targets[:,col]
    end

    df = DataFrame(; feature_columns..., target_columns...)

    obj = DataContainer(data_df=df,
                        n_samples=size(df,1),
                        n_features=n_features,
                        feature_names=names(df)[1:n_features],
                        n_targets=n_targets,
                        target_names=names(df)[end-n_targets:n_targets])

    shuffle_data && shuffle!(obj)

    return obj
end

# Replace this with something simpler.
DataContainer(cont::DataContainer, df::DataFrame) = DataContainer(df,
                                                                  n_samples=size(df,1),
                                                                  n_features=cont.n_features,
                                                                  feature_names=cont.feature_names,
                                                                  n_tragets=cont.n_targets,
                                                                  target_names=cont.target_names,
                                                                  shuffled=cont.shuffled,
                                                                  scales=cont.scales)

function storeDataAsDF(data_features, data_targets ; conv_one_hot=true)
    return df,n_features,n_targets
end

export extractArrays
function extractArrays(self::DataContainer)
    X = self.data_df[:, 1:self.n_features] |> Matrix
    y = self.data_df[:, self.n_features+1:end] |> Matrix
    return X, y
end

function shuffle!(self::DataContainer)
    self.data_df = self.data_df[randperm(self.n_samples),:]
    self.shuffled = true
end

export trainTestSplit
"""Split the dataset self.data_df into training and test sets"""
function trainTestSplit(self::DataContainer, frac=0.8, shuffle=true)
    shuffle!(self)

    n_train = round(Int, self.n_samples * frac)

    train_df = self.data_df[1:n_train, :]
    test_df = self.data_df[n_train+1:end, :]

    return DataContainer(self, train_df), DataContainer(self, test_df)
end
    
using RecipesBase
@recipe function plot(self::DataContainer)
    seriestype := :scatter
    markercolors := self.data_df[:,end]
    xlabel --> self.feature_names[1]
    ylabel --> self.feature_names[2]

    if self.n_features == 3
        zlabel --> self.feature_names[3]
    end

    # Bit of a mess! Does the necessary conversion from DataFrame to tuple with cols...
    self.data_df[:,1:end-1] |> Matrix |> eachcol |> x -> tuple(x...)
end

export scale
scale(self::DataContainer, X=self.data[1]) = scale!(self, similar(X), X)

export scale!
"""This is the equivalent of "inplace=true". """
function scale!(self::DataContainer, X_out=self.data[1], X=X_out)
    for feature_i = axes(X,2)
        arr = X[:,feature_i]

        the_mean = mean(arr)
        the_sd = std(arr)

        @. X_out[:,feature_i] = (arr - the_mean) / the_sd
    end

    return X_out
end

export addWhiteNoise!
function addWhiteNoise!(self::DataContainer, dist=Normal(0,1))
    feature_ind = 1 + count(x->startswith(String(x), "whitenoise"), self.feature_names)
    name = Symbol("whitenoise_$(feature_ind)")

    noise = rand(dist, self.n_samples)

    insertcols!(self.data_df, self.n_features+1, name => noise)
    push!(self.feature_names, name)
    push!(self.scales[1], 1)
    push!(self.scales[2], 0)
    self.n_features += 1
end


export oneHotEnc
function oneHotEnc(y)
    vals = sort(unique(y))
    # Going to be boring here
    # TODO allow general case
    # @argcheck vals == 1:length(vals)

    length(vals) > length(y)÷4 && error("Probably not correct input given $(length(vals)) unique values")

    out = zeros(Bool, length(y), length(vals))
    
    for (i,yi) in enumerate(y)
        out[i,findfirst(==(yi),vals)] = true
    end

    @argcheck all(sum(out,dims=2) .== 1)

    return out
end

export oneHotDec
oneHotDec(y, labels=axes(y,2)) = mapslices(row -> labels[argmax(row)], y, dims=2)
    # def back_transform(self, X_scaled=None, scales=None):
    #     """
    #     Back transform the scaled array to the untransformed original

    #     Parameters
    #     ----------

    #     Returns
    #     -------
    #     self : Restores unscaled input features
    #     """

    # def add_polynomial_features(self, degree=2, terms=None, powers_only=False, interaction_only=False, inplace=True):
    #     """
    #     Transform the original dataset X by adding polynomial and interaction features, e.g. X0**2, X1*X2

    #     Parameters
    #     ----------
    #     degree : int (default=2)
    #         The degree of polynomial and interaction terms to be added to the original dataset. For example, if the
    #         original dataset had three features: X0; X1; and X2 and we set the degree = 2, the new dataset would contain
    #         the additional features: X0**2, X1**2, and X2**2 and X0*X1, X0*X2 and X1*X2.
    #     terms : list (default = None)
    #         A list of the features to be transformed, i.e. if only a subset of features are of interest
    #     powers_only: Boolean (default = False)
    #         Include only polynomial terms, e.g. X0**3
    #     interaction_only: Boolean (default = False)
    #         Include only interaction terms, e.g. X0*X1
    #     in_place : Boolean (default = True)
    #         Store the modified (i.e. extended) array in place of the original dataset

    #     Returns
    #     -------
    #     X_new : array
    #         Extended array containing additional polynomial and interaction features
    #     """

end
