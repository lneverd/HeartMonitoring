#!/usr/bin/env python

# This file can be either used from the command line (type
# 'python jadeR.py --help' for usage, or see docstring of function main 
# below) or it can be imported as a module in a python shell or program 
# (use 'import jadeR').

# Comments in this source file are from the original MATLAB program, 
# unless they are preceded by 'GB'. This file has been updated from the 
# 1.8 version by fixing two typos that were spotted and corrected by
# David Rivest-Hénault from the National Research Council Canada


"""
jadeR

This module contains only one function, jadeR, which does blind source
separation of real signals. Hopefully more ICA algorithms will be added 
in the future.

jadeR requires NumPy.
"""

import sys
import os
import getopt
from numpy import abs, append, arange, arctan2, argsort, array, concatenate, \
    cos, diag, dot, eye, float64, loadtxt, matrix, multiply, ndarray, \
    savetxt, sign, sin, sqrt, zeros
from numpy.linalg import eig, pinv

__version__ = 1.8


def jadeR(X, m=None, verbose=False):
    """
    Blind separation of real signals with JADE.

    This function implements JADE, an Independent Component Analysis (ICA)
    algorithm developed by Jean-Francois Cardoso. More information about JADE
    can be found among others in: Cardoso, J. (1999) High-order contrasts for
    independent component analysis. Neural Computation, 11(1): 157-192.
    
    Translated into Numpy from the original Matlab Version 1.8 (May 2005) by
    Gabriel Beckers, http://gbeckers.nl . After that, two corrections were made
    by David Rivest-Hénault to make the code become equivalent at machine
    precision to that of jadeR.m

    Parameters:

        X -- an n x T data matrix (n sensors, T samples). Must be a NumPy array 
             or matrix.

        m -- number of independent components to extract. Output matrix B will 
             have size m x n so that only m sources are extracted. This is done 
             by restricting the operation of jadeR to the m first principal 
             components. Defaults to None, in which case m == n.

        verbose -- print info on progress. Default is False.

    Returns:

        An m*n matrix B (NumPy matrix type), such that Y = B * X are separated
        sources extracted from the n * T data matrix X. If m is omitted, B is a
        square n * n matrix (as many sources as sensors). The rows of B are
        ordered such that the columns of pinv(B) are in order of decreasing
        norm; this has the effect that the `most energetically significant`
        components appear first in the rows of Y = B * X.

    Quick notes (more at the end of this file):

    o This code is for REAL-valued signals.  A MATLAB implementation of JADE
        for both real and complex signals is also available from
        http://sig.enst.fr/~cardoso/stuff.html

    o This algorithm differs from the first released implementations of
        JADE in that it has been optimized to deal more efficiently
        1) with real signals (as opposed to complex)
        2) with the case when the ICA model does not necessarily hold.

    o There is a practical limit to the number of independent
        components that can be extracted with this implementation.  Note
        that the first step of JADE amounts to a PCA with dimensionality
        reduction from n to m (which defaults to n).  In practice m
        cannot be `very large` (more than 40, 50, 60... depending on
        available memory)

    o See more notes, references and revision history at the end of
        this file and more stuff on the WEB
        http://sig.enst.fr/~cardoso/stuff.html

    o For more info on NumPy translation, see the end of this file.

    o This code is supposed to do a good job!  Please report any
        problem relating to the NumPY code gabriel@gbeckers.nl

    Copyright original Matlab code: Jean-Francois Cardoso <cardoso@sig.enst.fr>
    Copyright Numpy translation: Gabriel Beckers <gabriel@gbeckers.nl>
    
    """

    # GB: we do some checking of the input arguments and copy data to new
    # variables to avoid messing with the original input. We also require double
    # precision (float64) and a numpy matrix type for X.

    assert isinstance(X, ndarray),\
           "X (input data matrix) is of the wrong type (%s)" % type(X)
    origtype = X.dtype # remember to return matrix B of the same type
    X = matrix(X.astype(float64))
    assert X.ndim == 2, "X has %d dimensions, should be 2" % X.ndim
    assert (verbose == True) or (verbose == False), \
           "verbose parameter should be either True or False"
    
    # GB: n is number of input signals, T is number of samples
    [n,T] = X.shape 
    assert n < T, "number of sensors must be smaller than number of samples"

    # Number of sources defaults to number of sensors
    if m==None:
        m=n 	
    assert m<=n,\
        "number of sources (%d) is larger than number of sensors (%d )" % (m,n)

    if verbose:
        print("jade -> Looking for %d sources" % m)
        print("jade -> Removing the mean value")
    X -= X.mean(1)


    # whitening & projection onto signal subspace
    # ===========================================

    if verbose: print("jade -> Whitening the data")
    # An eigen basis for the sample covariance matrix
    [D,U] = eig((X * X.T) / float(T))
    # Sort by increasing variances
    k = D.argsort()
    Ds = D[k]
    # The m most significant princip. comp. by decreasing variance
    PCs = arange(n-1, n-m-1, -1)

    # --- PCA  ----------------------------------------------------------
    # At this stage, B does the PCA on m components    
    B = U[:,k[PCs]].T

    # --- Scaling  ------------------------------------------------------
    # The scales of the principal components
    scales = sqrt(Ds[PCs])
    # Now, B does PCA followed by a rescaling = sphering
    B = diag(1./scales) * B
    # --- Sphering ------------------------------------------------------
    X = B * X 

    # We have done the easy part: B is a whitening matrix and X is white.
    
    del U, D, Ds, k, PCs, scales 

    # NOTE: At this stage, X is a PCA analysis in m components of the real
    # data, except that all its entries now have unit variance. Any further
    # rotation of X will preserve the property that X is a vector of
    # uncorrelated components. It remains to find the rotation matrix such
    # that the entries of X are not only uncorrelated but also `as independent
    # as possible". This independence is measured by correlations of order
    # higher than 2. We have defined such a measure of independence which 1)
    # is a reasonable approximation of the mutual information 2) can be
    # optimized by a `fast algorithm" This measure of independence also
    # corresponds to the `diagonality" of a set of cumulant matrices. The code
    # below finds the `missing rotation " as the matrix which best
    # diagonalizes a particular set of cumulant matrices.


    # Estimation of the cumulant matrices
    # ===================================

    if verbose: print("jade -> Estimating cumulant matrices")

    # Reshaping of the data, hoping to speed up things a little bit...
    X = X.T
    # Dim. of the space of real symm matrices
    dimsymm = (m * ( m + 1)) / 2
    # number of cumulant matrices
    nbcm = int(dimsymm)
    # Storage for cumulant matrices
    CM = matrix(zeros((m,m*nbcm), dtype=float64))
    R = matrix(eye(m, dtype=float64))
    # Temp for a cum. matrix
    Qij = matrix(zeros([m,m], dtype=float64))
    # Temp
    Xim	= zeros(m, dtype=float64)
    # Temp
    Xijm = zeros(m, dtype=float64)
    
    # I am using a symmetry trick to save storage. I should write a short note
    # one of these days explaining what is going on here.
    # will index the columns of CM where to store the cum. mats.
    Range = arange(m) 

    for im in range(m):
        Xim = X[:,im]
        Xijm = multiply(Xim, Xim)
        # Note to myself: the -R on next line can be removed: it does not affect
        # the joint diagonalization criterion
        Qij = multiply(Xijm, X).T * X / float(T) - R - 2 * (R[:,im] * R[:,im].T)
        CM[:,Range] = Qij 
        Range = Range  + m 
        for jm in range(im):
            Xijm = multiply(Xim, X[:,jm])
            Qij = sqrt(2) * (multiply(Xijm, X).T * X / float(T) 
                - R[:,im] * R[:,jm].T - R[:,jm] * R[:,im].T)
            CM[:,Range]	= Qij
            Range = Range + m

    # Now we have nbcm = m(m+1)/2 cumulants matrices stored in a big 
    # m x m*nbcm array.


    # Joint diagonalization of the cumulant matrices
    # ==============================================

    V = matrix(eye(m, dtype=float64))

    Diag = zeros(m, dtype=float64)
    On = 0.0
    Range = arange(m)
    for im in range(nbcm):
        Diag = diag(CM[:,Range])
        On = On + (Diag*Diag).sum(axis=0)
        Range = Range + m
    Off = (multiply(CM,CM).sum(axis=0)).sum(axis=1) - On
    # A statistically scaled threshold on `small" angles
    seuil = 1.0e-6 / sqrt(T)
    # sweep number
    encore = True 
    sweep = 0
    # Total number of rotations
    updates = 0
    # Number of rotations in a given seep
    upds = 0
    g = zeros([2,nbcm], dtype=float64)
    gg = zeros([2,2], dtype=float64)
    G = zeros([2,2], dtype=float64)
    c = 0
    s = 0
    ton	= 0
    toff = 0
    theta = 0
    Gain = 0

    # Joint diagonalization proper
    # ============================
    if verbose: print("jade -> Contrast optimization by joint diagonalization")

    while encore:
        encore = False
        if verbose: print("jade -> Sweep #%3d" % sweep)
        sweep = sweep + 1
        upds  = 0
        Vkeep = V

        for p in range(m-1):
            for q in range(p+1, m):

                Ip = arange(p, m*nbcm, m)
                Iq = arange(q, m*nbcm, m)

                # computation of Givens angle
                g = concatenate([CM[p,Ip] - CM[q,Iq], CM[p,Iq] + CM[q,Ip]])
                gg = dot(g, g.T)
                ton = gg[0,0] - gg[1,1] 
                toff = gg[0,1] + gg[1,0]
                theta = 0.5 * arctan2(toff, ton + sqrt(ton * ton + toff * toff))
                Gain = (sqrt(ton * ton + toff * toff) - ton) / 4.0

                # Givens update
                if abs(theta) > seuil:
                    encore = True
                    upds = upds + 1
                    c = cos(theta) 
                    s = sin(theta)
                    G = matrix([[c, -s] , [s, c] ])
                    pair = array([p,q])
                    V[:,pair] = V[:,pair] * G
                    CM[pair,:] = G.T * CM[pair,:]
                    CM[:,concatenate([Ip,Iq])] = \
                      append( c*CM[:,Ip]+s*CM[:,Iq], -s*CM[:,Ip]+c*CM[:,Iq], \
                              axis=1)
                    On = On + Gain
                    Off = Off - Gain

        if verbose: print("completed in %d rotations" % upds)
        updates = updates + upds
    
    if verbose: print("jade -> Total of %d Givens rotations" % updates)

    # A separating matrix
    # ===================
    B = V.T * B
    
    # Permute the rows of the separating matrix B to get the most energetic
    # components first. Here the **signals** are normalized to unit variance.
    # Therefore, the sort is according to the norm of the columns of
    # A = pinv(B)

    if verbose: print("jade -> Sorting the components")

    A = pinv(B)
    
    keys =  array(argsort(multiply(A,A).sum(axis=0)[0]))[0]
    B = B[keys,:]
    # % Is this smart ?
    B = B[::-1,:]

    if verbose: print("jade -> Fixing the signs")
    b = B[:,0]
    # just a trick to deal with sign == 0
    signs = array(sign(sign(b)+0.1).T)[0]
    B = diag(signs) * B

    return B.astype(origtype) 
