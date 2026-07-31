export trace, trace2, norm, normalize, dag, hermitianize, hermiticity, renyi2
export expect, expect1, expect2, expect_bond_bond, expect_1_bond
export entanglement_entropy, partial_trace, mutual_info_renyi2

function tensor_trace(state::State{Mixed}, i::Int)
    s = state.system
    j = SysIndex{Pure}(s, i)
    k = SysIndex{Mixed}(s, i)
    return dense(delta(j', j)) * combinerto(k, j', j)
end

tensor_obs(state::State{Pure}, ind::AtIndex{Pure, 1}) =
    tensor(state.system, ind)

function tensor_obs(state::State{Mixed}, ind::AtIndex{Pure, 1})
    s = state.system
    t = tensor(s, ind)
    j = SysIndex{Pure}(s, ind.index...)
    k = SysIndex{Mixed}(s, ind.index...)
    return t * combinerto(k, j, j')
end

tensor_obs(state::State{Mixed}, i::Int) =
    tensor_trace(state, i) * state.state[i]

function tensor_dag(state::State, i::Int)
    s = state.system
    j = SysIndex{Pure}(s, i)
    k = SysIndex{Mixed}(s, i)
    return dag(state.state[i]) * combinerto(k, j', j) * combinerto(k, j, j')
end

function get_loc(state::State, i::Int)
    l = state.preobs.loc
    if isempty(l)
        create_loc!(l, state)
    end
    return l[i]
end

function get_right(state::State, i::Int)
    r = state.preobs.right
    if isempty(r)
        create_right!(r, state)
    end
    return r[i]
end

function get_left(state::State, i::Int)
    l = state.preobs.left
    if length(l) < i
        create_left!(l, state, i)
    end
    return l[i]
end

"""
    trace(::State)

Return the trace of the system, mostly usefull for mixed representations.
This should be one.
"""
function trace(state::State)
    t = state.preobs.trace
    if isempty(t)
        create_trace!(t, state)
    end
    return t[1]
end

create_loc!(_, ::State{Pure}) = error("bug: get_loc on pure states")
function create_loc!(l, state::State{Mixed})
    n = length(state)
    resize!(l, n)
    for i in 1:n
        l[i] = tensor_obs(state, i)
    end
    return l
end

function create_right!(r, state::State{Pure})
    st = state.state
    s = state.system
    n = length(state)
    rl = ITensorMPS.rightlim(st) - 1
    resize!(r, n)
    r[n] = dag(st[n]')
    for i in n-1:-1:1
        rlink = commonind(st[i], st[i+1])
        v = if i >= rl
            delta(rlink, rlink')
        else
            k = SysIndex{Pure}(s, i+1)
            r[i+1] * delta(k, k') * st[i+1]
        end
        r[i] = v * dag(st[i]')
    end
    return r
end

function create_right!(r, state::State{Mixed})
    n = length(state)
    resize!(r, n)
    t = r[n] = ITensor(1.)
    for i in n-1:-1:1
        t = r[i] = get_loc(state, i+1) * t
    end
    return r
end

function create_trace!(t, state::State{Pure})
    resize!(t, 1)
    k = SysIndex{Pure}(state.system, 1)
    t[1] = scalar(get_right(state, 1) * delta(k, k') * state.state[1])
    return t
end

function create_trace!(t, state::State{Mixed})
    resize!(t, 1)
    t[1] = scalar(get_loc(state, 1) * get_right(state, 1))
    return t
end

function create_left!(l, state::State{Pure}, i::Int)
    st = state.state
    s = state.system
    ll = ITensorMPS.leftlim(st) + 1
    j = length(l)
    resize!(l, i)
    if j == 0
        l[1] = st[1] / real(trace(state))
        j = 1
    end
    for k in j+1:i
        llink = commonind(st[k-1], st[k])
        v = if k <= ll
            delta(llink, llink')
        else
            idx = SysIndex{Pure}(s, k-1)
            l[k-1] * delta(idx, idx') * dag(st[k-1]')
        end
        l[k] = v * st[k]
    end
    return l
end

function create_left!(l, state::State{Mixed}, i::Int)
    st = state.state
    j = length(l)
    resize!(l, i)
    if j == 0
        l[1] = st[1] / real(trace(state))
        j = 1
    end
    for k in j+1:i
        l[k] = l[k-1] * tensor_trace(state, k-1) * st[k]
    end
    return l
end

"""
    trace2(::State)

Return the trace of the square density matrix, mostly usefull for mixed representations.
This is one for pure representations.
"""
trace2(::State{Pure}) = 1.
trace2(state::State{Mixed}) = (norm(state.state) / real(trace(state))) ^ 2

"""
    norm(::State)

Return the norm of the state, mostly usefull for pure representations.
This should be one for pure representation.
"""
norm(state::State) = norm(state.state)


"""
    normalize(::State)

normalize state so that norm = 1 for pure state and trace = 1 for mixed state
"""
normalize(state::State{Pure}) =
    State(state, normalize(state.state))
normalize(state::State{Mixed}) =
    State(state, state.state / real(trace(state)))

"""
    dag(::State)

adjoint of density matrix for mixed representation
"""
dag(state::State{Pure}) =
    error("dag is meaningless on pure representations")
function dag(state::State{Mixed})
    n = length(state)
    st = MPS(n)
    for i in 1:n
        st[i] = tensor_dag(state, i)
    end
    State(state, st)
end


"""
    hermitianize(::State)

modify the state so that it is Hermitian (only useful for mixed state)
"""
hermitianize(state::State{Pure}; kwargs...) =
    state
hermitianize(state::State{Mixed}; limits::Limits=Limits()) =
    State(state, 0.5*(+(state.state, dag(state).state; limits.cutoff, limits.maxdim)))


"""
    hermiticity(::State)

hermiticity measure whether density matrix for mixed state is Hermitian as it should.

return a value from 0 (anti Hermitian) to 1 (Hermitian)

return 1 for pure state
"""
hermiticity(::State{Pure}) = 1.
hermiticity(state::State{Mixed}) =
    0.5 + 0.5 * real(dot(state.state, dag(state).state)) / norm(state.state)^2

"""
    renyi2(::State)
    renyi2(::State, ::Vector{Int})

renyi2 returns the Renyi entropy of order 2 of the state. This is 0. for pure representations.
When given an array of positions give the Renyi entropy of corresponding substate.
"""
renyi2(::State{Pure}) = 0.
renyi2(state::State{Mixed}) = -log(trace2(state))

renyi2(state::State{Mixed}, a::Vector{Int}) =
    renyi2(partial_trace(state, a; keepers = true))

unroll(x) =
    if x[1] isa Number
        x
    else
        v = [ map(t->t[i], x) for i in 1:length(x[1]) ]
        if x[1] isa Matrix
            return reshape(v, size(x[1]))
        else
            return v
        end
    end


struct Expector
    pos::Int
    t::ITensor
end

Expector() =
    Expector(0, ITensor())


zipto(state::State{Pure}, a::Expector, i::Int) = 
    if a.pos == 0
        Expector(i, get_left(state, i))
    elseif a.pos == i
        a
    else
        st = state.state
        t = a.t * dag(st[a.pos]')
        for k in a.pos+1:i-1
            idk = SysIndex{Pure}(state.system, k)
            t *= st[k] * delta(idk, idk')
            t *= dag(st[k]')
        end
        t *= st[i]
        Expector(i, t)
    end

zipto(state::State{Mixed}, a::Expector, i::Int) =
    if a.pos == 0
        Expector(i, get_left(state, i))
    elseif a.pos == i
        a
    else
        t = a.t
        for k in a.pos+1:i-1
            t *= get_loc(state, k)
        end
        t *= state.state[i]
        Expector(i, t)
    end

zipend(state::State, a::Expector) =
    Expector(a.pos, a.t * get_right(state, a.pos))


function expectfactor(state::State, a::Expector, o::AtIndex)
    a = zipto(state, a, o.index...)
    Expector(a.pos, a.t * tensor_obs(state, o))
end

expectfactor(state::State, a::Expector, o::Multi_F) =
    for i in o.start:o.stop
        a = expectfactor(state, a, F(i))
    end


function expect(state::State, coef::Number, subs::Vector{<:IndexedOp{Pure}})
    if coef == 0.
        return 0.
    end
    e = Expector()
    for o in subs
        e = expectfactor(state, e, o)
    end
    e = zipend(state, e)
    return coef * scalar(e.t)
end

"""
    expect(::State, obs)

Compute expectation values of `obs` on the given state.

# Examples
    expect(state, X(1)*Y(2) + Y(1)*Z(3))
    expect(state, [X(1)*Y(2), X(3), Z(1)*X(2)])

"""
expect(state::State, p::IndexedOp{Pure}) =
    expect(state, scalarcoef(p), prodsubs(p))
    
expect(state::State, op::SumOp{Pure, Indexed}) =
    sum(op.subs) do p
        expect(state, p)
    end

expect(state::State, op) =
    map(op) do o
        expect(state, o)
    end

expect1_one(state::State, op::SimpleOp, i::Int, t::ITensor) =
    if isfermionic(op)
        error("expect1 is not implemented for fermionic operators")
    else
        scalar(t * tensor_obs(state, op(i)))
    end

expect1_one(state::State, ops, i::Int, t::ITensor) =
    map(ops) do o
        expect1_one(state, o, i, t)
    end


"""
    expect1(::State, op)

Compute the expectation values of the given operators on all sites.

# Examples
    expect1(state, X)
    expect1(state, [X, Y, Z])

"""
function expect1(state::State, op)
    n = length(state)
    r = [ expect1_one(state, op, i, zipend(state, zipto(state, Expector(), i)).t) for i in 1:n ]
    return unroll(r)
end


function expect2(state::State, ops::Vector{<:Tuple{SimpleOp, SimpleOp}})
    oplist = [first.(ops) ; last.(ops)]
    need_fermionic = any(isfermionic, oplist)
    need_non_fermionic = any(x->!isfermionic(x), oplist)
    n = length(state)
    r = Matrix(undef, n, n)
    for i in 1:n
        t = zipend(state, zipto(state, Expector(), i)).t
        r[i, i] = map(ops) do (o1, o2)
            expect1_one(state, o1 * o2, i, t)
        end
        lnf = zipto(state, Expector(), i)
        lf = lnf
        for j in i+1:n
            enf = lnf
            ef = lf
            if need_non_fermionic
                enf = zipend(state, zipto(state, lnf, j))
            end
            if need_fermionic
                ef = zipend(state, zipto(state, lf, j))
            end
            r[i, j] = map(ops) do (o1, o2)
                if isfermionic(o1)
                    scalar(ef.t * tensor_obs(state, (o1 * F)(i)) * tensor_obs(state, o2(j)))
                else
                    scalar(enf.t * tensor_obs(state, o1(i)) * tensor_obs(state, o2(j)))
                end
            end
            r[j, i] = map(ops) do (o1, o2)
                if isfermionic(o1)
                    scalar(ef.t * tensor_obs(state, (o2 * F)(i)) * tensor_obs(state, o1(j)))
                else
                    scalar(enf.t * tensor_obs(state, o2(i)) * tensor_obs(state, o1(j)))
                end
            end
            if j < n
                if need_non_fermionic
                    lnf = zipto(state, expectfactor(state, lnf, Id(j)), j+1)
                end
                if need_fermionic
                    lf = zipto(state, expectfactor(state, lnf, F(j)), j+1)
                end
            end
        end
    end
    unroll(r)
end

"""
    expect2(::State, op_pairs)

Compute the 2-point correlations of the given pairs of operators on all sites.

# Examples
    expect2(state, (X, X))
    expect2(state, [(X, Y), (X, Z), (Y, Z)])
    ...
"""
expect2(state::State, ops::Tuple{SimpleOp, SimpleOp}) =
    expect2(state, [ops])[1]


# ── Three-site expectation value helper ──────────────────────────────────────
# Used internally by expect_bond_bond and expect_1_bond for overlapping cases.
function _expect3(state::State, oa::SimpleOp, ob::SimpleOp, oc::SimpleOp, a::Int, b::Int, c::Int)
    e = zipto(state, Expector(), a)
    e = Expector(a, e.t * tensor_obs(state, oa(a)))
    e = zipto(state, e, b)
    e = Expector(b, e.t * tensor_obs(state, ob(b)))
    e = zipto(state, e, c)
    e = Expector(c, e.t * tensor_obs(state, oc(c)))
    scalar(zipend(state, e).t)
end

"""
    expect_bond_bond(state, (op1, op2, op3, op4))

Compute ⟨op1(a) op2(a+1) op3(b) op4(b+1)⟩ for all bond pairs (a,b)
with 1 ≤ a,b ≤ N-1. Returns an (N-1)×(N-1) matrix.

Cases:
- |a-b| ≥ 2 : non-overlapping bonds, O(N²) sweep.
- a == b     : same bond, ⟨(op1·op3)(a) (op2·op4)(a+1)⟩.
- |a-b| == 1 : adjacent overlapping bonds, evaluated as a 3-site expectation.
"""
function expect_bond_bond(state::State, ops::NTuple{4, SimpleOp})
    o1, o2, o3, o4 = ops
    n = length(state)
    r = Matrix{ComplexF64}(undef, n-1, n-1)

    # ── Non-overlapping a < b (b ≥ a+2) ──────────────────────────────────────
    for a in 1:n-1
        # Bond-left env at a+1: o1(a) then o2(a+1) incorporated.
        la  = zipto(state, Expector(), a)
        la  = Expector(a,   la.t  * tensor_obs(state, o1(a)))
        la1 = zipto(state, la, a+1)
        la1 = Expector(a+1, la1.t * tensor_obs(state, o2(a+1)))

        a+2 > n-1 && continue
        sweep = zipto(state, la1, a+2)  # identity at a+2..a+1 (none), open ket at a+2

        for b in a+2:n-1
            # Apply o3(b), advance to b+1, apply o4(b+1), contract right.
            e = Expector(b,   sweep.t  * tensor_obs(state, o3(b)))
            e = zipto(state, e, b+1)
            e = Expector(b+1, e.t * tensor_obs(state, o4(b+1)))
            r[a, b] = scalar(e.t * get_right(state, b+1))

            # Advance sweep with identity at b (for next b).
            if b < n-1
                s     = Expector(b, sweep.t * tensor_obs(state, Id(b)))
                sweep = zipto(state, s, b+1)
            end
        end
    end

    # ── Non-overlapping a > b (a ≥ b+2): same sweep with roles swapped ───────
    for b in 1:n-1
        lb  = zipto(state, Expector(), b)
        lb  = Expector(b,   lb.t  * tensor_obs(state, o3(b)))
        lb1 = zipto(state, lb, b+1)
        lb1 = Expector(b+1, lb1.t * tensor_obs(state, o4(b+1)))

        b+2 > n-1 && continue
        sweep = zipto(state, lb1, b+2)

        for a in b+2:n-1
            e = Expector(a,   sweep.t  * tensor_obs(state, o1(a)))
            e = zipto(state, e, a+1)
            e = Expector(a+1, e.t * tensor_obs(state, o2(a+1)))
            r[a, b] = scalar(e.t * get_right(state, a+1))

            if a < n-1
                s     = Expector(a, sweep.t * tensor_obs(state, Id(a)))
                sweep = zipto(state, s, a+1)
            end
        end
    end

    # ── Diagonal a == b: ⟨(o1·o3)(a) (o2·o4)(a+1)⟩ ─────────────────────────
    for a in 1:n-1
        e = zipto(state, Expector(), a)
        e = Expector(a,   e.t  * tensor_obs(state, (o1*o3)(a)))
        e = zipto(state, e, a+1)
        e = Expector(a+1, e.t * tensor_obs(state, (o2*o4)(a+1)))
        r[a, a] = scalar(e.t * get_right(state, a+1))
    end

    # ── Adjacent a == b+1: overlap at site a, 3-site ⟨o3(b) (o1·o4)(a) o2(a+1)⟩ ──
    for a in 2:n-1
        b = a - 1
        r[a, b] = _expect3(state, o3, o1*o4, o2, b, a, a+1)
    end

    # ── Adjacent a == b-1: overlap at site a+1, 3-site ⟨o1(a) (o2·o3)(a+1) o4(b+1)⟩ ──
    for a in 1:n-2
        b = a + 1
        r[a, b] = _expect3(state, o1, o2*o3, o4, a, a+1, b+1)
    end

    return r
end


"""
    expect_1_bond(state, op_single, (op_bond_l, op_bond_r))

Compute ⟨op_single(i) · op_bond_l(j) op_bond_r(j+1)⟩ for all (i, j)
with 1 ≤ i ≤ N, 1 ≤ j ≤ N-1. Returns an N×(N-1) matrix.

Non-overlapping cases use O(N²) sweeps. Overlapping cases (i adjacent
to or coincident with the bond sites) are computed as 2- or 3-site
expectation values.
"""
function expect_1_bond(state::State, os::SimpleOp, bond_ops::NTuple{2, SimpleOp})
    ob1, ob2 = bond_ops
    n = length(state)
    r = Matrix{ComplexF64}(undef, n, n-1)

    # ── Single site i strictly left of bond (i ≤ j-2) ─────────────────────────
    # sweep_env carries l_{j-1} open (not k_{j-1}): os is already applied at i,
    # intermediate sites are traced with get_loc, and we open j via state.state[j].
    for i in 1:n-3  # need at least j = i+2 ≤ n-1, so i ≤ n-3
        li = zipto(state, Expector(), i)
        li = Expector(i, li.t * tensor_obs(state, os(i)))  # l_i open

        # Trace site i+1 to initialise sweep_env with l_{i+1} open.
        sweep_env = li.t * get_loc(state, i+1)

        for j in i+2:n-1
            t_j = sweep_env * state.state[j]   # l_{j-1} contracts; k_j, l_j open
            e   = Expector(j,   t_j * tensor_obs(state, ob1(j)))   # close k_j
            e   = zipto(state, e, j+1)
            e   = Expector(j+1, e.t * tensor_obs(state, ob2(j+1)))
            r[i, j] = scalar(zipend(state, e).t)

            # Advance: trace site j (reusing t_j), l_j open ready for j+1.
            sweep_env = t_j * tensor_trace(state, j)
        end
    end

    # ── Single site i strictly right of bond (i ≥ j+2) ───────────────────────
    for j in 1:n-1
        lj  = zipto(state, Expector(), j)
        lj  = Expector(j,   lj.t  * tensor_obs(state, ob1(j)))
        lj1 = zipto(state, lj, j+1)
        lj1 = Expector(j+1, lj1.t * tensor_obs(state, ob2(j+1)))

        j+2 > n && continue
        sweep = zipto(state, lj1, j+2)

        for i in j+2:n
            e = zipend(state, sweep)
            r[i, j] = scalar(e.t * tensor_obs(state, os(i)))

            if i < n
                s     = Expector(i, sweep.t * tensor_obs(state, Id(i)))
                sweep = zipto(state, s, i+1)
            end
        end
    end

    # ── i == j-1: 3-site ⟨os(i) ob1(j) ob2(j+1)⟩ ────────────────────────────
    for j in 2:n-1
        r[j-1, j] = _expect3(state, os, ob1, ob2, j-1, j, j+1)
    end

    # ── i == j:   2-site ⟨(os·ob1)(j) ob2(j+1)⟩ ─────────────────────────────
    for j in 1:n-1
        e = zipto(state, Expector(), j)
        e = Expector(j,   e.t  * tensor_obs(state, (os*ob1)(j)))
        e = zipto(state, e, j+1)
        e = Expector(j+1, e.t * tensor_obs(state, ob2(j+1)))
        r[j, j] = scalar(e.t * get_right(state, j+1))
    end

    # ── i == j+1: 2-site ⟨ob1(j) (ob2·os)(j+1)⟩ ─────────────────────────────
    for j in 1:n-1
        e = zipto(state, Expector(), j)
        e = Expector(j,   e.t  * tensor_obs(state, ob1(j)))
        e = zipto(state, e, j+1)
        e = Expector(j+1, e.t * tensor_obs(state, (ob2*os)(j+1)))
        r[j+1, j] = scalar(e.t * get_right(state, j+1))
    end

    return r
end


"""
    entanglement_entropy(::State, ::Int)

Return the entanglement entropy of the given state at the given site.
Also return the associated spectrum.
"""
function entanglement_entropy(state::State, pos::Int)
    s = orthogonalize(state.state, pos)
    _, S = svd(s[pos], (linkinds(s, pos-1)..., siteinds(s, pos)...))
    sp = [ S[i,i]^2 for i in 1:dim(S, 1) ]
    sp /= sum(sp)
    ee = -sum(p * log(p) for p in sp)
    return (ee, sp)
end

"""
    partial_trace(::State, ::Vector{Int} [; keepers = true])

return the state partially traced at the given positions,
alternatively one can give the positions to keep by setting `keepers = true`
"""
function partial_trace(state::State{Mixed}, pos::Vector{Int}; keepers::Bool = false)
    n = length(state)
    if keepers
        keep = pos
    else
        keep = filter(e->e ∉ pos, 1:n)
    end
    kn = length(keep)
    if kn == 0
        error("partial_trace cannot trace all sites of a state")
    end
    mps = state.state
    sys = state.system
    j = 0
    t = Vector{ITensor}(undef, kn)
    s = Vector{AbstractSite}(undef, kn)
    sp = Vector{Index}(undef, kn)
    sm = Vector{Index}(undef, kn)
    for (i, k) in enumerate(keep)
        s[i] = sys[k]
        sp[i] = SysIndex{Pure}(sys, k)
        sm[i] = SysIndex{Mixed}(sys, k)
        if i == 1
            t[1] = copy(get_left(state, k))
        else
            for l in j+1:k-1
                t[i-1] *= get_loc(state, l)
            end 
            t[i] = copy(mps[k])
        end
        j = k
    end
    t[kn] *= get_right(state, keep[kn])
    for i in 1:kn-1
        idx = commonind(t[i], t[i+1])
        jdx = settags(idx, "Link, l=$i")
        replaceind!(t[i], idx, jdx)
        replaceind!(t[i+1], idx, jdx)
    end
    return State{Mixed}(System(s, sp, sm), MPS(t))
end

"""
    mutual_info_renyi2(state::State, cut::Int)
    mutual_info_renyi2(state::State, a::Vector{Int})

return an approximation of the mutual information using renyi2 entropy.
You define the two parts either by giving the position of the cut between the left and right parts or by giving the list of positions for one of the parts.
"""
mutual_info_renyi2(state::State, a::Vector{Int}) =
    renyi2(partial_trace(state, a; keepers = true)) +
    renyi2(partial_trace(state, a; keepers = false)) -
    renyi2(state)

mutual_info_renyi2(state::State, cut::Int) =
    mutual_info_renyi2(state, collect(1:cut))