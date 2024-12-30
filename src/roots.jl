# An implementation of Brent's method, translated from the algorithm described in
#   https://en.wikipedia.org/wiki/Brent%27s_method

# The reference description provides no guidance on the stopping criteria, so we choose to
# use both relative and absolute tolerances (similar to `isapprox()`).

function brent(f, a::T, b::T; abstol = nothing, reltol = nothing) where {T}
    δ = isnothing(abstol) ? eps(abs(b - a)) : T(abstol)
    ε = isnothing(reltol) ? eps(abs(b - a)) : T(reltol)
    fa = f(a)
    fb = f(b)
    if fa * fb ≥ zero(T)
        # not a bracketing interval
        return oftype(a, NaN)
    end
    if abs(fa) < abs(fb)
        b, a = a, b
        fb, fa = fa, fb
    end

    c, fc = a, fa
    d = s = b
    mflag = true

    while true
        Δ = abs(b - a)
        if iszero(fb) || Δ <= δ || Δ <= abs(b) * ε
            # converged
            return b
        end

        if fa != fc && fb != fc
            # inverse quadratic interpolation
            s = ( a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb)))
        else
            # secant
            s = b - fb * (b - a) / (fb - fa)
        end

        u, v = (3a + b) / 4, b
        if u > v
            u, v = v, u
        end
        tol = max(δ, max(abs(b), abs(c), abs(d)) * ε)

        cond1 = !(u < s < v)
        cond23 = abs(s - b) ≥ abs(mflag ? b - c : c - d) / 2
        cond45 = abs(mflag ? b - c : c - d) < tol
        if cond1 || cond23 || cond45
            # bisection
            s = (a + b) / 2
            mflag = true
        else
            mflag = false
        end
        fs = f(s)
        if iszero(fs)
            return s
        end

        c, fc, d = b, fb, c
        if sign(fa) * sign(fs) < zero(T)
            b, fb = s, fs
        else
            a, fa = s, fs
        end

        if abs(fa) < abs(fb)
            b, a = a, b
            fb, fa = fa, fb
        end
    end
end
