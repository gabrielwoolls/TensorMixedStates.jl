"""
Potts.jl — q=3 transverse-field clock/Potts site type for TensorMixedStates.

To activate, add to the TMS fork's src/TensorMixedStates.jl:
    include("Potts.jl")
(place it alongside the Qubits.jl, Spins.jl include lines).

Basis: |0⟩, |1⟩, |2⟩  (Eta-clock eigenstates, eigenvalues 1, ω, ω²)

Hamiltonian (KZ sweep convention matching TFI_Github/Sim_Potts/):
    H(t) = -J(t) Σᵢ (η†ᵢηᵢ₊₁ + h.c.) - h(t) Σᵢ (τᵢ + τ†ᵢ)
    J(t) = v·t,  h(t) = 2 − v·t
"""

export Potts_mod

const ω3 = exp(2π * im / 3)

"""
    struct Potts <: AbstractSite

Three-level (qutrit) site for the q=3 clock/Potts model.

States:
- "0", "1", "2"  — Eta-clock eigenbasis states
- "Plus"         — (|0⟩+|1⟩+|2⟩)/√3, eigenstate of (Tau+Tau†) with eigenvalue +2

Operators (exported via the Potts_mod submodule):
- Eta    : clock operator diag(1, ω, ω²), ω = e^{2πi/3}   (unitary, non-Hermitian)
- Tau    : cyclic shift |j⟩→|j+1 mod 3⟩                   (unitary, non-Hermitian)
- NumQ   : number operator diag(0,1,2)                      (Hermitian)
- Proj0, Proj1, Proj2 : projectors |a⟩⟨a|                  (Hermitian)
"""
struct Potts <: AbstractSite end

dim(::Potts) = 3

@def_states(Potts(),
[
"0"    => [1., 0., 0.],
"1"    => [0., 1., 0.],
"2"    => [0., 0., 1.],
"Plus" => [1., 1., 1.] / √3,
])

@def_operators(Potts(),
[
selfadjoint_op =>
[
NumQ          = [0. 0. 0.; 0. 1. 0.; 0. 0. 2.],
Proj0         = [1. 0. 0.; 0. 0. 0.; 0. 0. 0.],
Proj1         = [0. 0. 0.; 0. 1. 0.; 0. 0. 0.],
Proj2         = [0. 0. 0.; 0. 0. 0.; 0. 0. 1.],
EtaHermSum = [2. 0. 0.; 0. -1. 0.; 0. 0. -1.],            # η+η†, real order parameter
TauHermSum = [0. 1. 1.; 1. 0. 1.; 1. 1. 0.],              # τ+τ†, Hermitian field op
EtaHermDiff   = [0. 0. 0.; 0. -√3 0.; 0. 0. √3],             # i(η-η†) = diag(0,-√3,√3)
TauHermDiff   = ComplexF64[0 im -im; -im 0 im; im -im 0],     # i(τ-τ†)
],
plain_op =>
[
Eta          = [1. 0. 0.; 0. ω3 0.; 0. 0. conj(ω3)],
EtaDag       = [1. 0. 0.; 0. conj(ω3) 0.; 0. 0. ω3],
Tau          = [0. 0. 1.; 1. 0. 0.; 0. 1. 0.],
TauDag       = [0. 1. 0.; 0. 0. 1.; 1. 0. 0.],
Zeta         = [0. 0. 1.; conj(ω3) 0. 0.; 0. ω3 0.],    # ζ=η†τ
ZetaPrime    = [0. 1. 0.; 0. 0. conj(ω3); ω3 0. 0.],    # ζ′=η†τ†
],
])

"""
Potts_mod

Submodule re-exporting Potts site operators.  Use as:

    using TensorMixedStates, .Potts_mod

Exports: Potts, Eta, Tau, NumQ, Proj0, Proj1, Proj2
"""
module Potts_mod
    import ..Potts, ..Eta, ..EtaDag, ..EtaHermSum, ..EtaHermDiff,
           ..Tau, ..TauDag, ..TauHermSum, ..TauHermDiff,
           ..Zeta, ..ZetaPrime,
           ..NumQ, ..Proj0, ..Proj1, ..Proj2
    export Potts, Eta, EtaDag, EtaHermSum, EtaHermDiff,
           Tau, TauDag, TauHermSum, TauHermDiff,
           Zeta, ZetaPrime,
           NumQ, Proj0, Proj1, Proj2
end
