# SinglePileAnalysis

## Theory

There are many methodologies to analyze pile behavior:
1-Empirical methods: mainly some huge amount of analysis were done, and a correlation is extracted from them
2-Analytical methods: mainly are some closed form solutions that model the behavior of pile with a continuum elastic space that have very strict assumptions
3-Numerical methods: These methods have a very huge computation cost, but the nonlinear plastic behavior of soil can be simulated.

In this study finite element analysis (FEA), that is a numerical method is implemented using Opensees library. Soil behavior is modeled using nonlinear springs.
There are many abnormalities and parameters that are very difficult to model, like construction methodologies, soil layer abnormalities, concrete shrinkage and so on. To simulate these parameters and other uncertain parameters, some parameters are embedded in the soil model to simulate these parameters.
To calibrate these parameters an optimization algorithm is used to calibrate these parameter with the pile axial load test.
In this study differential evolution algorithm is used as optimization algorithm to minimize the error of numerical method with pile load test result.

## Test

Install python packages
`python -m pip install -r requirements.txt`

Running tests
`pytest`
