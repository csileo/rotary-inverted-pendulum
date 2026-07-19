import MeshCat

import ForwardDiff as FD
import RigidBodyDynamics as RBD

function integrate_rk4(f::Function, vars::Vector{T}, dt::AbstractFloat)::Vector{T} where {T}
    x = vars[1:4]
    u = vars[5:5]

    f1 = dt * f([x; u])
    f2 = dt * f([x + f1 / 2; u])
    f3 = dt * f([x + f2 / 2; u])
    f4 = dt * f([x + f3; u])

    return x + (1 / 6) * (f1 + 2 * f2 + 2 * f3 + f4)
end

"""
Linearise the dynamics of the rotary inverted pendulum.
Returns the state-transition matrix A and the control (or input) matrix B.
"""
function linearise_dynamics(
    mechanism::RBD.Mechanism, dt::T;
    xgoal=T[0, π, 0, 0], ugoal=T[0],
) where {T<:AbstractFloat}
    cache_state = RBD.StateCache(mechanism)
    cache_dynamicsresult = RBD.DynamicsResultCache(mechanism)

    function rip_dynamics(vars::Vector{T}) where {T}
        state = cache_state[T]
        dynamicsresult = cache_dynamicsresult[T]

        # Unpack variables
        q = vars[1:2]
        v = vars[3:4]
        u = vars[5:5]

        # Set the state of the system
        RBD.set_configuration!(state, q)
        RBD.set_velocity!(state, v)

        # the 2nd dof is not actuated
        torques = [u; zero(T)]

        # Calculate the forward dynamics
        RBD.dynamics!(dynamicsresult, state, torques)
        v̇ = dynamicsresult.v̇

        return [v; v̇]
    end

    # `A` is the state-transition matrix. We get it by linearising the discretised
    # version of our model's full nonlinear dynamics with respect to the _state_.
    # `B` is the input or control matrix. We get it by linearising the discretised
    # version of our model's full nonlinear dynamics with respect to the _input_.
    A = FD.jacobian(dx -> integrate_rk4(rip_dynamics, [dx; ugoal], dt), xgoal)
    B = FD.jacobian(du -> integrate_rk4(rip_dynamics, [xgoal; du], dt), ugoal)

    A = convert(Matrix{T}, A)
    B = convert(Matrix{T}, B)

    return rip_dynamics, A, B
end
