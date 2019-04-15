Random weights for classification.


# Setup

    using Revise
    using Plots
    pyplot()
    
    push!(LOAD_PATH, "../src")
    using Data
    using LogBin


## Generic tester

    function TestData(data)
        cont = DataContainer(data)
        X,y = extractArrays(cont)
    
        class = LogisticClassifierBinary()
        initialiseWeights!(class, X)
    
        plotFit(class, X, y)
    end


# Sprial

    TestData(makeSpiral(n_samples=1000))

![img](images/rand_weights_Spiral.png)

    TestData(makeCloud())

![img](images/rand_weights_Cloud.png)

    TestData(makeDonut())

![img](images/rand_weights_Donut.png)

    TestData(makeXor())

![img](images/rand_weights_Xor.png)

    TestData(makeMoons())

![img](images/rand_weights_Moons.png)

