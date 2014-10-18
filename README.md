This is my clone of [SimpleLDA-R](https://github.com/cjrd/SimpleLDA-R). I will use this code as a groudtruth to check with my Julia implementation. 

Below is the original Readme file from SimpleLDA-R

=================

This code provides a simple and straightforward (and slow!) implementation of variational expectation maximization inference for latent Dirchlet allocation. Check https://github.com/cjrd/SimpleLDA-R to make sure you have the most up-to-date version of this code and the accompanying tutorial.

Author: Colorado Reed (colorado . j . reed At gmail . com)

You are free to use this code and tutorial in anyway that you see fit (I encourage reproducing this code in your own fashion and discourage plagiarism).

NOTE: This tutorial is largely based on the original LDA paper by David Blei, Andrew Ng, and Michael Jordan: http://www.cs.berkeley.edu/~blei/papers/blei03a.pdf


lda-tutorial-reed.pdf: contains a conversational tutorial on latent Dirchlet allocation, along with a full pseudocode implementation. The R code in this file directly draws from this pseudocode.

cololda.R The implementation of the LDA inference method discussed above. The document is commented to aid readability and encourage the interested reader to work through the actual LDA implementation---convince yourself that LDA isn't magic!

auxfunctions.R Auxiliary functions that have been removed from the main script (cololda.R) to improve readability. Source these functions before sourcing cololda.R

data: folder contains a formatted corpus of 2246 documents from the Associated Press -- acquired from Dave Blei: http://www.cs.princeton.edu/~blei/lda-c/index.html
