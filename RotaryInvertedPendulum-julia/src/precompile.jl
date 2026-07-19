import RotaryInvertedPendulum as RIP

using PrecompileTools: @setup_workload, @compile_workload

# PrecompileTools allows you to reduce the latency of the first execution of Julia code.
# It is applicable for package developers and for "ordinary users" in their personal workflows.
# See the GitHub repository and docs for more details: https://github.com/JuliaLang/PrecompileTools.jl.

@setup_workload begin

    # Putting some things in `@setup_workload` instead of `@compile_workload` can
    # reduce the size of the precompile file and potentially make loading faster.

    # Load the URDF model of the rotary inverted pendulum
    package_path = joinpath(pkgdir(RIP), "..")
    filename = joinpath(package_path, "urdf/model.urdf")
    mechanism = RBD.parse_urdf(filename)

    @compile_workload begin

        # All calls in this block will be precompiled, regardless of whether they belong to this package or not.

        rip_dynamics, A, B = RIP.linearise_dynamics(mechanism, 1 / 150)

    end

end
