# Proof-of-concept ideal ballooning stability boundary using Gaussian Processes

This is a demonstration of finding the ideal ballooning stability boundary with
the gyrokinetic solver GS2 and Gaussian Process optimisation (GPO).

There are two Python scripts that each run 25 GS2 simulations across the same
set of parameters to find the ideal ballooning stability
boundary. `scan_with_pk.py` uses Pyrokinetics to do a 5x5 uniform scan over `s`
and $\alpha$, while `scan_with_gp.py` uses GPO from Inference Tools to do the
same scan in a more flexible way.

WARNINGS:

- this is just a demonstration of the techniques, and not necessarily the best
  way to achieve the results with these tools;
- the simulations are almost certainly massively underresolved in order for the
  whole scan to run quickly, and no physical interpretations should be made
  about the results;
- at the very least, you will need to change the path to `gs2`
