# Notes

These are notes for an initial prototype experiment for using the generalised profile likelihood method on ion channel models and data.

There are only a few iterative exploratory steps recorded in the git history, so it may be hard to look through and find ideas.

## First problem: Data-driven time-dependent voltage
First attempt was to simply write down the equations and blast them into the library [pypei](https://github.com/dwu402/pypei) that I had written for fitting the Samoa measles data. Defaults aside, the first major tripping point was pulling out the data-driven voltage information.

The library is written on top of casadi, an auto-differentiation language/library, which has (at least) two classes of symbolics: SX and MX. SX symbolics are explicit in that each element of a vector is represented separately, so it is fast, but (more) memory hungry. MX symbolics are more implicit, and are in general used to represent vecotrs matrices etc.: they are less memory intensive, but are slow as mollasses. These two types of symbolics do not mix, so you kind of have to commit to one or the other. There are ways to get around this in particular cases: MX symbolics can be `expand`ed into SX symbolics sometimes, and you can also just pre-evaluate some things if you need (which I do for the basis).

Interpolation in casadi are in MX symbolics. The library is written so that it uses SX symbolics.
I tried two ways of coping with this:
1. the `process_volts_again.py` constructs a string that represents the staircase function using piecewise `if_else` calls. I think I got a dimensionality thing wrong at some point, but my library complains about this for some reason.
2. I pre-'discretised' the voltage. This means that the data must also be at these voltage values (or a subset thereof), or the model will not be able to compute the correspodning state.

## Second problem: Basis functions
Once we got past the voltage problem, we run into the most enduring problem: how do we specify a basis that both:
- captures the behaviour of the model (for example the fast dynamics of the r state and the slower/smoother dynamics of the a state)
- imposes enough regularity in the solution:
    - for smoothness (removing noise)
    - and for derivative evaluation later

[B splines](https://en.wikipedia.org/wiki/B-spline):

are defined by knots. Repeating knots means that you can have discontinuties in the higher-order derivatives. Additionally repeating knots allows you to have lower-order discontinuities. For cubic splines, having 4 repeated knots causes discontinuities at that knot location.
This may be good for capturing the overall behaviour (jumps at voltage drops), but I think it breaks some of the nice properties of the continuity of derivatives that is required for decent model fits.

To riff on this a bit, the formulation we use imposes a 'weak' constraint on the model, since we are not solving it exactly (via numerical integration). This means that as long as the collocation points are aligned somewhat along the vector field at _some_ point, the discrepancy is low. The smoothness of the B-spline does a double duty by trying to ensure that the solution is also somewhat "smooth" and thus continuous, which 'solves' the weakness of the model constraint.
With discontinuities we lose this continuity requirement, so the states effectively can 'reset' at each of these 4-repeat knot locations which is somewhat annoying for the physical accuracy of the estimates.

I also explored the fourier basis a bit on some non-ion channel synthetic data in the explore_basis_functions notebook, but I'm not quite convinced that that is a good idea, for the same reasont that discontinuous B-splines are a good idea.

No harm trying either though.

The last thing to think about is that I assume that all the states are on the same basis. This is not necessary, but would require a lot of rejigging of things in the `modeller` submodule (or rejigging of a few things but then a lot of thinking about the resulting consequences).
This might be able to resolve the problems of the difference in speeds of the two species.

## Third problem: Identifiability

We see from the formulation of the current equation that we 'observe' the product of a and r.
We know from experiments that the dynamics of r are faster than a. This actually yields the robust behaviour we can extract from the current data.
However, the formulation of the ODE does not impose any restrictions on the speed of the dynamics. So we run into situations where the estimates found are both middling compromise solutions that aren't necessarily physical. Kinda frustrating. For example, we do some very minimal regularisation in both notbooks where we actually do some fitting (`att1_local` and `single_jump_calibration`) and neither are successful. I didn't necessarily want to go very heavy-handed.
g also doesn't help, since it can encourage very `degenerate` solutions (in the single_jump notebook).

## Fourth problem: Traditional methods also don't seem great

naive_mle has been runnig for about ~1.5~ 2 hours now and shows no sign of convergence.

## Fifth problem: pypei is not documented

Not even going to try to say it's not _well_-documented. It's pretty opaque.

On the other hand, there are other libraries that might be worth thinking about:

- Collocinfer (R) is the OG generalised profiling method by Ramsay and Hooker.
- all-at-once methods in the geophysics community probably are implemented somewhere and are mathematically identical to my approach. I can't find their code immediately, but Haber and Ascher (2001) might be a starting point.
        
- JAX is a nice autodiff library - though google does move fast with their projects, so there might be a different one by now
- we use CasADi, which is written mainly for the optimal control field
- autodiff is also an autodiff library
- machine learning is heavy into ther autodifferentiation (and reverse-mode/adjoint methods), so they're bound to have other nice libraries out there
