'''newton.py

Implementation of a Newton-Raphson root-finder.

'''

import numpy as np
import functions as F

class Newton(object):
    """Newton objects have a solve() method for finding roots of f(x)
    using Newton's method. x and f can both be vector-valued.

    """
    
    def __init__(self, f, tol=1.e-6, maxiter=20, dx=1.e-6, **kwargs):
        """Parameters:
        
        f: the function whose roots we seek. Can be scalar- or
        vector-valued. If the latter, must return an object of the
        same "shape" as its input x

        tol, maxiter: iterate until ||f(x)|| < tol or until you
        perform maxiter iterations, whichever happens first

        dx: step size for computing approximate Jacobian

        """
        for key,value in kwargs.items():
            if key == 'max_radius':
                self._max_radius = value
                
            if key == 'Df':
                self._Df = value

        if f.__class__.__name__ == 'Polynomial':
            self._Df = f.differentiate()

        self._f = f
        self._tol = tol
        self._maxiter = maxiter
        self._dx = dx

    def solve(self, x0):
        """Determine a solution of f(x) = 0, using Newton's method, starting
        from initial guess x0. x0 must be scalar or 1D-array-like. If
        x0 is scalar, return a scalar result; if x0 is 1D, return a 1D
        numpy array.

        """
        # NOTE: no need to check whether x0 is scalar or vector. All
        # the functions/methods invoked inside solve() return "the
        # right thing" when x0 is scalar.
        x = x0
        conv = 0
        for i in range(self._maxiter):
            fx = self._f(x)
            # linalg.norm works fine on scalar inputs
            if np.linalg.norm(fx) < self._tol:
                conv = 1
                return x
            x = self.step(x, fx)
            # check to see if |xk - x0| > max_radius & raise exception if so
            if (hasattr(self, '_max_radius')):
                if (np.linalg.norm(x - x0) > self._max_radius):
                    mesg1 = "Calculated root exceeded maximum radius "
                    mesg2 = "from initial input"
                    mesg = mesg1 + mesg2
                    raise RuntimeError(mesg)
                else:
                    return x

        if conv == 0:
            raise RuntimeError("Solution did not converge within maximum number of iterations")
        else:
            return x

    def step(self, x, fx=None):
        """Take a single step of a Newton method, starting from x. If the
        argument fx is provided, assumes fx = f(x).

        """
        if fx is None:
            fx = self._f(x)

        # Find the derivative of f at x:
        # Check to see if analytical jacobian has been provided:
        if (hasattr(self, '_Df')): #Analytical method provided
            Df_x = self._Df(x)
        else: #Use the approximate Method
            Df_x = F.approximateJacobian(self._f, x, self._dx)

        
        if np.all(Df_x==0):
            counter = self._maxiter
            while np.all(Df_x==0):
                if counter == 0:
                    raise RuntimeError("Was not able to find a non-zero slope/jacobian near the provided x0")
                x = x + self._dx
                Df_x = F.approximateJacobian(self._f, x, self._dx)
                counter = counter - 1
        
        # linalg.solve(A,B) returns the matrix solution to AX = B, so
        # it gives (A^{-1}) B. np.matrix() promotes scalars to 1x1
        # matrices.
        h = np.linalg.solve(np.matrix(Df_x), np.matrix(fx))
        # Suppose x was a scalar. At this point, h is a 1x1 matrix. If
        # we want to return a scalar value for our next guess, we need
        # to re-scalarize h before combining it with our previous
        # x. The function np.asscalar() will act on a numpy array or
        # matrix that has only a single data element inside and return
        # that element as a scalar.
        if np.isscalar(x):
            h = np.asscalar(h)
        else:
            h = np.asarray(h)
        
        return x - h
