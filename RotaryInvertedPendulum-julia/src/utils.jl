using Plots

function play_trajectory_animation(mvis, q_mat, dt)
    ts = range(0, step=dt, length=size(q_mat, 2))
    q_arr = [q_mat[:, i] for i in axes(q_mat, 2)]
    animation = MeshCat.Animation(mvis, ts, q_arr)
    MeshCat.setanimation!(mvis, animation)
end

function plot_trajectory(positions::Matrix, velocities::Matrix, torques::Matrix)
    p1 = plot(title="positions")
    p2 = plot(title="velocities")
    p3 = plot(title="torques")

    hline!(p1, [π], color=:red, label=nothing, linestyle=:dash, linewidth=1)
    plot!(p1, positions[1, :], color=1, label="sm", linewidth=2)
    plot!(p1, positions[2, :], color=2, label="p", linewidth=2)

    plot!(p2, velocities[1, :], color=1, label="sm", linewidth=2)
    plot!(p2, velocities[2, :], color=2, label="p", linewidth=2)

    plot!(p3, torques[1, :], color=1, label="sm", linewidth=2)

    plot(
        p1, p2, p3,
        layout=grid(1, 3),
        legend=:topright,
        size=(700, 230),
    )
end
