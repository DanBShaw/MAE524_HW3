#!/usr/bin/env python3

import unittest
import numpy as np
import numpy.testing as npt

import newton

class TestNewton(unittest.TestCase):
    def testLinear(self):
        # Just so you see it at least once, this is the lambda keyword
        # in Python, which allows you to create anonymous functions
        # "on the fly". As I commented in testFunctions.py, you can
        # define regular functions inside other
        # functions/methods. lambda expressions are just syntactic
        # sugar for that.  In other words, the line below is
        # *completely equivalent* under the hood to:
        #
        # def f(x):
        #     return 3.0*x + 6.0
        #
        # No difference.
        f = lambda x : 3.0*x + 6.0

        # Setting maxiter to 2 b/c we're guessing the actual root
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(-2.0)
        # Equality should be exact if we supply *the* root, ergo
        # assertEqual rather than assertAlmostEqual
        self.assertEqual(x, -2.0)

    def test_ConvergenceException(self):
        # Test whether or not an exception is raised because no solution
        # was obtained after the maximum number of iterations
        
        # Create an f with no root:
        f = lambda x : x**2 + 1.

        # Create Solver
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)

        with self.assertRaises(RuntimeError) as context:
            solver.solve(2.0)

        self.assertTrue("Solution did not converge within maximum number of iterations" in str(context.exception))

    def test_0Slope(self):
        # Test whether an exception is raised when the slope is 0
        # near the provided x value
        
        # Create an f with zero slope:
        f = lambda x : 1.
            
        # Create Solver
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)

        with self.assertRaises(RuntimeError) as context:
            solver.solve(2.0)

        self.assertTrue("Was not able to find a non-zero slope/jacobian near the provided x0" in str(context.exception))

    def test_commonf_1(self):
        # Tests f = (x+1)(x-1)

        # Create an f with zero slope:
        f = lambda x : (x+1.)*(x-1.)

        # Create Solver
        solver = newton.Newton(f, tol=1.e-15, maxiter=50)

        x_neg1 = solver.solve(-0.5)
        x_1 = solver.solve(0.5)

        self.assertAlmostEqual(x_neg1, -1.0)
        self.assertAlmostEqual(x_1, 1.0)


    def test_commonf_2(self):
        # Tests f = (x+1)*(x-1)*100
        # f has a very steep slope around the origin

        # Create an f with zero slope:
        f = lambda x : 100*(x+1.)*(x-1.)

        # Create Solver
        solver = newton.Newton(f, tol=1.e-15, maxiter=50)

        x_neg1 = solver.solve(-2)
        x_1 = solver.solve(0.6)

        self.assertAlmostEqual(x_neg1, -1.0)
        self.assertAlmostEqual(x_1, 1.0)

    def test_common2d_f(self):
        # Tests f = (x+1)*(x-1)*100
        # f has a very steep slope around the origin

        # Create an f with zero slope:
        f = lambda x : np.array([[1-x[0,0]],[-x[1,0]]])

        # Create Solver
        solver = newton.Newton(f, tol=1.e-15, maxiter=50)

        x0 = np.array([[1],[1]])
        x_CalcSol = solver.solve(x0)
        x_Sol = np.array([[1],[0.]])

        npt.assert_array_almost_equal(x_CalcSol, x_Sol)
        
        
if __name__ == "__main__":
    unittest.main()

    
