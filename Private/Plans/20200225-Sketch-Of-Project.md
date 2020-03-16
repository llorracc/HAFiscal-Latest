# Preliminary Description of the Project

'Conventional' monetary policy is simply the manipulation of the short-run nominal interest rate.
This simplicity has made it possible to make rapid progress in HA modeling of monetary policy.

One reason that there are few HA papers on fiscal policy is that there are almost infinitely many different policies that could be described as 'fiscal.' As a result, there is little if anything that can be generically said about the effects of 'fiscal policy.' 

The idea for this paper is to make a start on the analysis of fiscal policy with HA models by constructing a detailed analysis of what the models say about the effects of two specific fiscal stimulus policies that were both undertaken in the U.S. during the Great Recession, precisely for the purpose of stimulating aggregate demand.

The two policies were:

1. In 2008, one-time lump-sum `stimulus checks` were mailed to households
1. In 2009-10, there was a cut in the wage tax that finances Social Security and related transfer programs
	
While there are now rich literatures on the partial equilibrium effects of these policies, to date no paper has provided a comprehensive analysis of their effectiveness in a modeling framework that is consistent with the broad features of micro and macro consumption evidence. 

The idea for this paper is to use a model like that of [Sticky Expectations and Consumption Dynamics](http://econ.jhu.edu/people/ccarroll/papers/cAndCwithStickyE), which _is_ consistent with both micro and macro consumption data, to analyze the model's implications for the effectiveness of these two policies.

A first step will be to use the Norwegian lottery data to recalibrate the model of the StickyE paper so that

1. The distribution of time preference rates is chosen in such a way as to match the distribution of MPCs
1. The structure of income shocks is calibrated to match facts from the Norwegian registry data

## Matching which MPC?

My suggestion is that we calibrate the model to match the amount by which expenditures are higher two years after receipt of the lottery winnings
   * This gets around the problem that first-year MPE's are much too large to be explained in the off-the-shelf model of nondurable expenditures
   * We can then come up with some simple specification that captures whatever happens in the first year (like, everyone expends 30 percent off the top in the form of purchasing durable goods (including 'memories of that time I won the lottery' that people purchase by having a party for neighbors, buying durable goods, taking extravagant vacations, etc

## Evaluating the Dynamics

We can start with the version of the model published in the StickyE paper, recalibrated to match the distribution of MPC's (as above). 

Work is in progress to add to the HARK toolkit the capacity to analyze New Keynesian models as well as `real` models. That will take some time, but should be available by the time we need to calculate the dynamics of aggregate demand.
