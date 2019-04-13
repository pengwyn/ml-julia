
# Table of Contents

1.  [Donut](#org4f12530)
2.  [Cloud](#org15d1cf1)
3.  [Spiral](#org2dfa5c5)
4.  [Xor](#org6796fcf)
    1.  [2D](#org4f74a5e)
    2.  [3D](#org5c9caa7)
5.  [Moons](#orgd145a6b)

    union!(LOAD_PATH, ["../src"])
    using Data


<a id="org4f12530"></a>

# Donut

    data = makeDonut()
    container = DataContainer(data)
    plot(container)

![img](images/donut.png)


<a id="org15d1cf1"></a>

# Cloud

    data = makeCloud()
    container = DataContainer(data)
    plot(container)

![img](images/cloud.png)


<a id="org2dfa5c5"></a>

# Spiral

    data = makeSpiral(4, n_samples=5000)
    container = DataContainer(data)
    plot(container)

![img](images/spiral.png)


<a id="org6796fcf"></a>

# Xor


<a id="org4f74a5e"></a>

## 2D

    data = makeXor(n_samples=1000)
    container = DataContainer(data)
    plot(container)

![img](images/xor.png)


<a id="org5c9caa7"></a>

## 3D

    data = makeXor(3, n_samples=1000)
    container = DataContainer(data)
    plot(container)

![img](images/xor_3d.png)


<a id="orgd145a6b"></a>

# Moons

    data = makeMoons()
    container = DataContainer(data)
    plot(container)

![img](images/moons.png)

