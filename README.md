# MAE524_HW3

The Newton class attempts to find the root of a provided function with Newton-Raphson techniques

Newton(Function, tol=1.e-15, maxiter=2, max_radius=1, Df=Function_Derivative)

INPUTS:
f: function of which the roots are searched

tol: (optionally provided - if not, default = 1.e-6) the tolerance at which |f(x)| will be considered equal to 0

maxiter: (optionally provided - if not, default = 20) the maximum number of iterations that the script will perform in an attempt to find a root before raising an exception

dx: (optionally provided - if not, default = 1.e-6) the step size for computing the approximate Jacobian (only used if an analytical jacobian description isn't given)

max_radius: (optionally provided) A bounding for x such that if x is ever greater than the provided estimate, an exception is raised

Df: (optionally provided) An function that returns the analytical derivative for f


OUTPUTS:
x: the root that was found

or an exception is raised because of an error

------------------------------------------------------------------------
Function Description
f(x) is evaluated, and if it's not within a prescribed tolerance, a new x is calculated by Newton-Raphson techniques (symmetric difference quotient) or by a provided analytical derivative function.

If a root is not found after a maxiter number of iterations, an exception is thrown:
RuntimeError: Solution did not converge within maximum number of iterations

If a root is calculated to be more than the max_radius away from the initially provided root, an exception is throw:
RuntimeError: Calculated root exceeded maximum radius from initial input

If the derivative is ever calculated to be zero at any point x, the derivative of f(x+dx) is returned instead. If the derivative is still zero after repeating this process for the maximum number of iterations, an exception is raised:
RuntimeError: Was not able to find a non-zero slope/jacobian near the provided x0

A polynomial type function can be input as the function to the Newton method. If it is, the analytical derivative is calculated and then automatically storedas the analytical derivate so that the Newton Raphson method doesn't have to be used.

------------------------------------------------------------------------
Examples:

f = lambda x : x - 1.
solver = newton.Newton(f, tol=1.e-15, maxiter=10)
solver.solve(2.0)

The solver will return the root: x ~= 1 (approximately - within rounding error)