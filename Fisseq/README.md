# IARPA Data Processing

This document countains a readme detailing processing for IARPA data.

# Info

* Channel mapping is GTAC, in order.

# Tasks

* Build proper calling algorithm
* Seperate functions, and do a covering streamlining function.
* 

# Seperation

The programs should be seperated into these categories:

* TIF reader. Have this, works well.
* Region divider. Adapt sergeys implementation to a script. (Maybe a region reader)
* Background removal. Simple gaussian subtraction.
* Bootstrap for puncta fitting (generic, but fits gaussian)
* Watershed regions (contains blurring)
* Discriminator, to remove bad points
* Calling algorithm (bootstraps data, uses multidiscriminator, and runs ML)

