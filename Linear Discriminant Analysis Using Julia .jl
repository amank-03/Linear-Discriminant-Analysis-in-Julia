# importing libraries 

using LinearAlgebra
using Statistics
using RDatasets
using BenchmarkTools
using CSV
using DataFrames
using Plots

# _class_cov 

function _class_cov(X , y, priors)
```
    Computes within-class covariance matrix
    
    input: 
    X : Predictor matrix 
    y: target values 
    priors: class priors
        
    output:
    cov_ : Weighted within-class covariance matrix
    
```
    classes = sort(unique(y))
    cov_ = zeros(size(X)[2], size(X)[2])
    
    for (idx, group) in enumerate(classes)
        Xg = X[y .== group, :]
        cov_ += (priors[idx])*(cov(Matrix(Xg)))
    end
    return(cov_)
end

# mean_col

function mean_col(X)
```
    computes mean of each column 
    
    input:
    X: Input data 
    
    output:
    mean_vec: mean value of each column
    
```
    mean_vec = [mean(c) for c in eachcol(X)]
    return(mean_vec)
end

# _class_Mean

function _class_Mean(X,y)
```
    Computes class mean 
    
    input: 
    X: Input data
    y: target values
    
    output
    
    mean_ :  column wise mean for each class"(a matrix of shape(number of attributes , number of classes))"
    
```
    classes = sort(unique(y))
    
    mean_ = zeros(size(X)[2] ,length(classes) )
    for (idx, group) in enumerate(classes)
        Xg = X[y .== group, :]
        mean_val = [mean(c) for c in eachcol(Xg)]
        mean_[: ,idx] = mean_val
    end
    return(mean_)
end

# _priors 

function _priors(X,y)
```
    Computes class priors 
    
    input: 
    X: Input data
    y: target values
    
    output:
    prior_ :  prior value for each class

```

    classes = sort(unique(y))
    
    prior_ = zeros(length(classes) )
    for (idx, group) in enumerate(classes)
        Xg = X[y .== group, :]
        prop = size(Xg)[1]/size(X)[1]
        prior_[idx] = prop
    end
    return(prior_)
end

function _make_cat(y)
```
    transform target variable into numerical categorical varible 
    
    input:
    y: target variable
    
    output:
    my_arr: numerical values corresponding to each categorical varible"(1:number of classes)"
    
```
    classes = sort(unique(y))
    
    labels = [x for x in range(1,length(classes))]
    
    my_arr = [0 for x in range(1, length(y))]
    
    for (id ,val) in enumerate(y)
        for (idx, class) in enumerate(classes)
            if val == class
                my_arr[id] = idx
            end
        end
    end
    return(my_arr)
end

# _solve_lsqr
```
    Computes a straightforward solution of the optimal decision rule based directly on the discriminant functions.
       using formula
    coef_ = inv"(sigma)""*"_class_Mean
    
    Note: dimensionality reduction is not supported with this solver
    
    input: 
    X : Predictor matrix 
    y: target values 
    priors: class priors
        
    output:
    coef_ : coefficient of discriminant functions
    intercept_ :  intercept of discriminant functions
    
    
```


function _solve_lsqr(X,y , priors::Union{Vector, Nothing}=nothing)
    if priors !== nothing
        @assert round(sum(priors); digits=4) == 1
    end
    
    if priors === nothing 
        priors = _priors(X,y)
    end
    
    mean_ = _class_Mean(X,y)
    cov_ = _class_cov(X , y, priors)
    ln_priors = [log(prior) for prior in priors]
    
    coef_ = inv(cov_)*mean_
    intercept_ =  -0.5 * diag(transpose(mean_)*coef_) + ln_priors
    
    return(coef_, intercept_)
end


function _solve_eigen(X,y ,priors::Union{Vector, Nothing}=nothing )

```
    Computes the optimal solution of the Rayleigh coefficient"(J(W))", "(basically the ratio of 
    between class scatter to within class scatter)" 
    "J(W) = \frac{W.T*Sb*W}{W.T*Sw*W}"
    
    This solver supports both classification and dimensionality reduction
    
    input: 
    X : Predictor matrix 
    y: target values 
    priors: class priors
        
    output:
    coef_ : coefficient of discriminant functions
    intercept_ :  intercept of discriminant functions
       
```
    
    if priors !== nothing
        @assert round(sum(priors); digits=4) == 1
    end
    
    if priors === nothing 
        priors = _priors(X,y)
    end
    
    mean_ = _class_Mean(X,y)
    cov_ = _class_cov(X , y, priors)
    ln_priors = [log(prior) for prior in priors]
    
    Sw = cov_  # within scatter
    St = cov(Matrix(X))  # total scatter
    Sb = St - Sw
    
    evals , evecs = eigen(Sb, Sw)
    
    coef_ = (transpose(mean_)*evecs)*transpose(evecs)
    intercept_ =  -0.5 * diag(transpose(mean_)*transpose(coef_)) + ln_priors
    
    return(coef_, intercept_)
end

function _solve_SVD(X,y , tol ,priors::Union{Vector, Nothing}=nothing)
```
    SVD solver
    
    This solver supports both classification and dimensionality reduction
    
    input: 
    X : Predictor matrix 
    y: target values 
    tol: tolerance value for cut-off eigen value of diag matrix of SVD 
    priors: class priors
    
    output:
    pramas:
        intercept: intercept values of discriminant functions 
        coef: coef of discriminant functions
        rank: rank 
        scalings: transformation matrix"(W)"
    
    
```
    
    if priors !== nothing
        @assert round(sum(priors); digits=4) == 1
    end
    
    if priors === nothing 
        priors = _priors(X,y)
    end
    
    n_samples = size(X)[1]
    classes = sort(unique(y))
    mean_ = _class_Mean(X,y)
    cov_ = _class_cov(X , y, priors)
    ln_priors = [log(prior) for prior in priors]


    Xc = []
    for (idx, group) in enumerate(classes)
        Xg = X[y .== group, :]
        Xg_mean = broadcast(- , Xg , transpose(mean_[:,idx]))
        push!(Xc , Matrix(Xg_mean))
    end 

    xbar_ = mean_col(X)

    Xc = vcat(Xc...)

    std_ =  std(Xc , dims=1)
    std_[std_.==0] .= 1

    fac = 1.0 / (size(X)[1] - length(classes))

    X_temp = sqrt(fac)*broadcast(/, Xc , std_)

    U, S, Vt = svd(X_temp)

    S_diag = S[S.>=tol]
    rank = length(S_diag)
    scalings_temp = broadcast(/, Vt[:, 1:rank] , transpose(std_))
    scalings = broadcast(/, scalings_temp , transpose(S[1:rank]))

    n_classes = length(classes)
    fac_2 = ifelse.( n_classes==1 , 1, 1/(n_classes-1))
    temp_1 = [sqrt(val) for val in (n_samples * priors *fac_2) ]
    temp_2 = broadcast(-, mean_ , xbar_)
    temp_3 = broadcast(* , transpose(temp_2) , temp_1)

    X_new = temp_3*scalings

    U, S, Vt = svd(X_new, full=false)

    rank = length(S[S.>=tol])
    scalings_ = scalings*Vt[:, 1:rank]

    coef = transpose(temp_2)*scalings_

    intercept_ = (-0.5)*sum(coef.*coef , dims=2) + ln_priors
    coef_ = coef*transpose(scalings_)
    intercept_ = intercept_ - coef_*xbar_
    
    
    if length(classes) ==2
        coef_ = coef_[2,:] - coef_[1,:]
        intercept_ = intercept_[2] -  intercept_[1]
    end
    
    params = Dict("intercept" => intercept_ , "coef" => coef_, "rank" => rank, "scalings" => scalings_)

    return(params)
end







function _solve_moment(X,y , tol ,priors::Union{Vector, Nothing}=nothing)

```
    SVD solver, uses standard estimators of the mean and variance "(replicating R's solution)"
    
    This solver supports both classification and dimensionality reduction
    
    input: 
    X : Predictor matrix 
    y: target values 
    tol: tolerance value for cut-off eigen value of diag matrix of SVD 
    priors: class priors
    
    output:
    pramas:
        intercept: intercept values of discriminant functions 
        coef: coef of discriminant functions
        rank: rank 
        scalings:  transformation matrix"(W)"
```
    
    if priors !== nothing
        @assert round(sum(priors); digits=4) == 1
    end
    
    if priors === nothing 
        priors = _priors(X,y)
    end
    
    n_samples = size(X)[1]
    classes = sort(unique(y))
    mean_ = _class_Mean(X,y)
    cov_ = _class_cov(X , y, priors)
    ln_priors = [log(prior) for prior in priors]

    Xc = []
    for (idx, group) in enumerate(classes)
        Xg = X[y .== group, :]
        Xg_mean = broadcast(- , Xg , transpose(mean_[:,idx]))
        push!(Xc , Matrix(Xg_mean))
    end

    xbar_ = mean_col(X)

    Xc = vcat(Xc...)

    std_ =  std(Xc , dims=1)
    std_[std_.==0] .= 1

    fac = 1.0 / (size(X)[1] - length(classes))
    X_temp = sqrt(fac)*broadcast(/, Xc , std_)
    
    
    U, S, Vt = svd(X_temp)
    S_diag = S[S.>=tol]
    rank = length(S_diag)
    scalings_temp = broadcast(/, Vt[:, 1:rank] , transpose(std_))
    scalings = broadcast(/, scalings_temp , transpose(S[1:rank]))

    n_classes = length(classes)
    fac_2 = ifelse.( n_classes==1 , 1, 1/(n_classes-1))

    temp_1 = [sqrt(val) for val in (n_samples * priors *fac_2) ]
    temp_2 = broadcast(-, mean_ , xbar_)
    temp_3 = broadcast(* , transpose(temp_2) , temp_1)

    X_new = temp_3*scalings

    U, S, Vt = svd(X_new, full=false)
    rank  = length(S[S.>=tol])
    scalings_ = scalings*Vt[:, 1:rank]
    intercept_temp_1 = transpose(temp_2)*scalings_
    intercept_ = (-0.5)*sum(intercept_temp_1.*intercept_temp_1 , dims=2) + ln_priors
    intercept_temp_2 = intercept_temp_1*transpose(scalings_)
    intercept_ = intercept_ - intercept_temp_2*xbar_
    params = Dict("intercept" => intercept_ , "coef" => scalings_, "rank" => rank, "scalings" => scalings_)

    return(params)
end


iris = dataset("datasets", "iris")

X = iris[:,1:4]

labels = [0 for y in 1:150]
for i = 1:size(iris, 1)
    x = iris[:, 5][i]
    if x == "setosa"
        labels[i] = 1
    elseif x == "virginica"
        labels[i] = 2
    else
        labels[i] = 3
    end
end

y = labels;

intercept, coef = _solve_eigen(X,y)
intercept

tol = 0.0001
paramas_iris = _solve_SVD(X,y , tol)
paramas_iris["coef"]

tol = 0.0001
@benchmark pramas = _solve_SVD(X,y , tol)

proj = pramas["coef"]

make_normal(transpose(proj))

pramas["intercept"]

@benchmark p = _solve_moment(X,y , 0.0001)

params_R_iris = _solve_moment(X,y ,
    0.0001)
params_R_iris["coef"]

params_R_iris = _solve_moment(X,y , 0.0001)
proj = params_R_iris["coef"]
make_normal(proj)

function make_normal(X)
    
    temp = X.*X
    temp = sum(temp , dims = 1)
    temp = .√(temp)
    temp = broadcast(/ , X , temp)
    return(temp)
end

make_normal(proj)

using CSV
using DataFrames

df = CSV.read("penguins_lter.csv" , DataFrame)
df = df[!,["Culmen Length (mm)" , "Culmen Depth (mm)", "Flipper Length (mm)","Body Mass (g)","Sex" ]]
df = dropmissing(df::AbstractDataFrame)
y = df[!,"Sex"]
labels = zeros(length(y))
for (idx,sex) in enumerate(y)
    if sex =="MALE"
        labels[idx] = 1
    end
end
y = labels
X = df[!,["Culmen Length (mm)" , "Culmen Depth (mm)", "Flipper Length (mm)","Body Mass (g)"]];


df = CSV.read("file1.csv" , DataFrame)

X = df[!,["Culmen Length (mm)" , "Culmen Depth (mm)", "Flipper Length (mm)","Body Mass (g)"]]

y = df[!,"Species"]
labels = zeros(length(y))
for (idx,specie) in enumerate(y)
    if specie =="Adelie Penguin (Pygoscelis adeliae)"
        labels[idx] = 1
    elseif specie == "Gentoo penguin (Pygoscelis papua)"
        labels[idx] = 2
    end
end
y = labels;


params_R_penguin = _solve_moment(X,y,
    0.0001 )
params_R_penguin["coef"]

params_R_penguin["coef"]

make_normal(params_R["coef"])


4×2 Matrix{Float64}:
  0.0820607   -0.998165
 -0.991605    -0.052959
  0.0999184    0.0290805
  0.00133546   0.00413342


4×2 Matrix{Float64}:
  0.084554    -0.998213
 -0.992998    -0.0501766
  0.0824825    0.0321884
  0.00124401   0.00408829


Coefficients of linear discriminants:
                            LD1          LD2
Culmen.Length..mm.   0.08832666 -0.417870885
Culmen.Depth..mm.   -1.03730494 -0.021004854
Flipper.Length..mm.  0.08616282  0.013474680
Body.Mass..g.        0.00129952  0.001711436


Coefficients of linear discriminants:
                            LD1          LD2
Culmen.Length..mm.   0.08832666 -0.417870885
Culmen.Depth..mm.   -1.03730494 -0.021004854
Flipper.Length..mm.  0.08616282  0.013474680
Body.Mass..g.        0.00129952  0.001711436

P

params_R = _solve_moment(X,y , 0.0001 )

tol = 0.0001
pramas_py_penguin = _solve_SVD(X,y , tol)
pramas_py_penguin["coef"]

make_normal(pramas_py["coef"])

array([[ 1.06214785e+00,  2.06466324e+00, -2.06101568e-01,
        -7.55752064e-03],
       [-7.55773193e-01,  3.38788801e+00, -2.68350550e-01,
        -2.36789997e-03],
       [ 3.40615434e-01, -5.30055439e+00,  4.43380810e-01,
         7.08507560e-03]])

df = CSV.read("file2.csv" , DataFrame)

X = df[!,["Culmen Length (mm)" , "Culmen Depth (mm)", "Flipper Length (mm)","Body Mass (g)"]]

y = df[!,"Sex"]

labels = zeros(length(y))
for (idx,sex) in enumerate(y)
    if sex =="MALE"
        labels[idx] = 1
    end
end
y = labels;


#params = _solve_SVD(X,y , 0.0001 )

@benchmark pramas = _solve_SVD(X,y , tol)

params["coef"]

pramas_py["rank"]

pramas_py["intercept"]

mpg = dataset("ggplot2", "mpg")

X = mpg[!,["Displ", "Cyl"]]
y = _make_cat(mpg[!,"Drv"])


@benchmark pramas_mpg = _solve_moment(X,y , 0.0000001 )

pramas_mpg["coef"]
