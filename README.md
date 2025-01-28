# mfccx

MFCC implementation in JAX. 

## Upstream

[python_speech_features](https://github.com/jameslyons/python_speech_features)

### Be careful about this issue

**The pip version of the upstream warehouse is outdated code.**

We noticed that the MFCC results obtained by using the package installed via pip do not match those obtained by running the warehouse source code. '

After checking the code, we found that the discrepancy is due to the code in pip being inconsistent with the warehouse source code. 

We believe that the warehouse source code is the more accurate version (pip was last updated on August 16, 2017, while the last commit to the warehouse was on December 31, 2020). 

Therefore, we will base our numerical stability on the results provided by the warehouse source code.

## denpendencies
`numpy scipy soundfile`