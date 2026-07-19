#!/usr/bin/env julia
#
# Data collection and visualization script for PIDControl
#
# Usage:
#   julia --project=./RotaryInvertedPendulum-julia ./RotaryInvertedPendulum-arduino/PIDControl/collect_and_plot.jl <port> [duration_seconds]
#
# Example:
#   julia --project=./RotaryInvertedPendulum-julia ./RotaryInvertedPendulum-arduino/PIDControl/collect_and_plot.jl /dev/cu.usbserial-10 10
#
# The script will:
#   1. Connect to the Arduino running PIDControl
#   2. Enable data printing (send 'P')
#   3. Collect data for the specified duration
#   4. Disable data printing (send 'P')
#   5. Parse and plot the data
#   6. Save plots as PNG images in the same directory
#

using LibSerialPort
using Plots
using Dates
using Statistics

const BAUD_RATE = 500000
const STEPS_PER_REVOLUTION = 200 * 8  # 1600 steps/rev with 8x microstepping
const DEGREES_PER_STEP = 360.0 / STEPS_PER_REVOLUTION

function steps_to_degrees(steps::Number)
    return steps * DEGREES_PER_STEP
end

function collect_data(port_name::String, duration_seconds::Int)
    println("Opening serial port: $port_name at $BAUD_RATE baud")

    data_lines = String[]

    LibSerialPort.open(port_name, BAUD_RATE) do port
        # Wait for Arduino to reset and boot (takes ~2 seconds)
        println("Waiting for Arduino to boot...")
        sleep(2.5)

        # Flush any boot messages
        while LibSerialPort.bytesavailable(port) > 0
            LibSerialPort.read(port, UInt8)
        end

        # Enable printing
        println("Enabling data output (sending 'P')...")
        LibSerialPort.write(port, [UInt8('P')])
        sleep(0.2)  # Give Arduino time to process

        # Debug: check if we're receiving anything
        println("Checking for data stream...")
        sleep(0.5)
        bytes_available = LibSerialPort.bytesavailable(port)
        println("  Bytes available: $bytes_available")

        if bytes_available == 0
            println("  No data yet - this is normal if pendulum is not near vertical")
            println("  Data output only occurs during BALANCING state")
        end

        println("\nCollecting data for $duration_seconds seconds...")
        println("TIP: Move the pendulum close to vertical to start balancing!")

        start_time = time()
        buffer = UInt8[]

        while (time() - start_time) < duration_seconds
            if LibSerialPort.bytesavailable(port) > 0
                byte = LibSerialPort.read(port, UInt8)
                if byte == UInt8('\n')
                    if !isempty(buffer)
                        line = String(copy(buffer))
                        push!(data_lines, line)
                        empty!(buffer)
                    end
                elseif byte != UInt8('\r')
                    push!(buffer, byte)
                end
            else
                sleep(0.001)  # Small sleep to prevent busy-waiting
            end

            # Progress indicator every second
            elapsed = time() - start_time
            if length(data_lines) > 0 && length(data_lines) % 100 == 0
                print("\rCollected $(length(data_lines)) samples ($(round(elapsed, digits=1))s)...")
            end
        end

        println("\nDisabling data output (sending 'P')...")
        LibSerialPort.write(port, [UInt8('P')])
        sleep(0.1)
    end

    println("Collected $(length(data_lines)) lines of data")
    return data_lines
end

struct ParsedData
    timestamps::Vector{Float64}
    motor_positions::Vector{Float64}
    motor_targets::Vector{Float64}
    pendulum_angles::Vector{Float64}
    states::Vector{Int}
    p_terms::Vector{Float64}
    i_terms::Vector{Float64}
    d_terms::Vector{Float64}
    loop_freqs::Vector{Float64}
    overruns::Vector{Int}
end

function parse_data(lines::Vector{String})
    timestamps = Float64[]
    motor_positions = Float64[]
    motor_targets = Float64[]
    pendulum_angles = Float64[]
    states = Int[]
    p_terms = Float64[]
    i_terms = Float64[]
    d_terms = Float64[]
    loop_freqs = Float64[]
    overruns_vec = Int[]

    t = 0.0
    dt = 0.010  # 10ms between samples (100 Hz)

    for line in lines
        parts = split(strip(line), ',')
        ncols = length(parts)

        # Support formats: 3+ columns
        if ncols < 3
            continue
        end

        try
            motor_pos = parse(Float64, parts[1])
            motor_target = parse(Float64, parts[2])
            # Pendulum and PID terms are sent as ×1000 integers for speed
            pendulum_deg = parse(Float64, parts[3]) / 1000.0
            state_val = ncols >= 4 ? parse(Int, parts[4]) : 1
            p_term = ncols >= 5 ? parse(Float64, parts[5]) / 1000.0 : 0.0
            i_term = ncols >= 6 ? parse(Float64, parts[6]) / 1000.0 : 0.0
            d_term = ncols >= 7 ? parse(Float64, parts[7]) / 1000.0 : 0.0
            loop_freq = ncols >= 8 ? parse(Float64, parts[8]) : 0.0
            overruns_val = ncols >= 9 ? parse(Int, parts[9]) : 0

            push!(timestamps, t)
            push!(motor_positions, steps_to_degrees(motor_pos))
            push!(motor_targets, steps_to_degrees(motor_target))
            push!(pendulum_angles, pendulum_deg)
            push!(states, state_val)
            push!(p_terms, p_term)
            push!(i_terms, i_term)
            push!(d_terms, d_term)
            push!(loop_freqs, loop_freq)
            push!(overruns_vec, overruns_val)
            t += dt
        catch e
            continue
        end
    end

    return ParsedData(timestamps, motor_positions, motor_targets, pendulum_angles,
                      states, p_terms, i_terms, d_terms, loop_freqs, overruns_vec)
end

function compute_statistics(pendulum_angles::Vector{Float64}, motor_positions::Vector{Float64})
    # Find the upright target (should be around ±180°)
    # The target is the closest multiple of 180° to the mean
    mean_angle = mean(pendulum_angles)
    target = round(mean_angle / 180) * 180

    errors = pendulum_angles .- target

    stats = Dict(
        "target_angle" => target,
        "mean_error" => mean(errors),
        "std_error" => std(errors),
        "max_error" => maximum(abs.(errors)),
        "rms_error" => sqrt(mean(errors.^2)),
        "motor_range" => maximum(motor_positions) - minimum(motor_positions),
        "samples" => length(pendulum_angles),
        "duration" => length(pendulum_angles) * 0.01
    )

    return stats
end

function create_plots(data::ParsedData, output_dir::String)
    # Ensure output directory exists
    mkpath(output_dir)

    plots_created = String[]

    timestamps = data.timestamps
    pendulum_angles = data.pendulum_angles
    motor_positions = data.motor_positions
    motor_targets = data.motor_targets

    # Determine the upright target
    mean_angle = mean(pendulum_angles)
    target_angle = round(mean_angle / 180) * 180

    # Plot 1: Pendulum angle over time
    p1 = plot(timestamps, pendulum_angles,
        label="Actual",
        xlabel="Time (s)",
        ylabel="Pendulum Angle (°)",
        title="Pendulum Angle vs Time",
        linewidth=1,
        legend=:topright
    )
    hline!(p1, [target_angle], label="Target ($(target_angle)°)", linestyle=:dash, color=:red)

    filename1 = joinpath(output_dir, "plot_pendulum_angle.png")
    savefig(p1, filename1)
    push!(plots_created, filename1)
    println("Saved: $filename1")

    # Plot 2: Pendulum error over time
    errors = pendulum_angles .- target_angle
    p2 = plot(timestamps, errors,
        label="Error",
        xlabel="Time (s)",
        ylabel="Error (°)",
        title="Pendulum Error (Actual - Target)",
        linewidth=1,
        color=:red
    )
    hline!(p2, [0], label="Zero", linestyle=:dash, color=:black)

    filename2 = joinpath(output_dir, "plot_pendulum_error.png")
    savefig(p2, filename2)
    push!(plots_created, filename2)
    println("Saved: $filename2")

    # Plot 3: Motor position over time
    p3 = plot(timestamps, motor_positions,
        label="Actual Position",
        xlabel="Time (s)",
        ylabel="Motor Position (°)",
        title="Motor Position vs Time",
        linewidth=1
    )
    plot!(p3, timestamps, motor_targets,
        label="Target Position",
        linestyle=:dash,
        linewidth=1
    )

    filename3 = joinpath(output_dir, "plot_motor_position.png")
    savefig(p3, filename3)
    push!(plots_created, filename3)
    println("Saved: $filename3")

    # Plot 4: PID terms over time (new diagnostic plot)
    has_pid_data = any(x -> x != 0, data.p_terms) || any(x -> x != 0, data.i_terms)
    if has_pid_data
        p4 = plot(timestamps, data.p_terms, label="P term", linewidth=1)
        plot!(p4, timestamps, data.i_terms, label="I term", linewidth=1)
        plot!(p4, timestamps, data.d_terms, label="D term", linewidth=1)
        plot!(p4, timestamps, data.p_terms .+ data.i_terms .+ data.d_terms,
            label="Total", linewidth=2, color=:black, linestyle=:dash)
        xlabel!(p4, "Time (s)")
        ylabel!(p4, "PID Output (°)")
        title!(p4, "PID Terms Over Time")

        filename4 = joinpath(output_dir, "plot_pid_terms.png")
        savefig(p4, filename4)
        push!(plots_created, filename4)
        println("Saved: $filename4")
    end

    # Plot 5: Combined overview (2x2 subplot)
    p5 = plot(
        plot(timestamps, pendulum_angles, ylabel="Pendulum (°)", xlabel="", title="Pendulum Angle", legend=false),
        plot(timestamps, errors, ylabel="Error (°)", xlabel="", title="Pendulum Error", legend=false, color=:red),
        plot(timestamps, motor_positions, ylabel="Motor (°)", xlabel="Time (s)", title="Motor Position", legend=false),
        plot(errors, motor_positions, xlabel="Pendulum Error (°)", ylabel="Motor Position (°)", title="Error vs Motor", legend=false, seriestype=:scatter, markersize=1, alpha=0.3),
        layout=(2,2),
        size=(1000, 800)
    )

    filename5 = joinpath(output_dir, "plot_overview.png")
    savefig(p5, filename5)
    push!(plots_created, filename5)
    println("Saved: $filename5")

    return plots_created
end

function main()
    if length(ARGS) < 1
        println("Usage: julia collect_and_plot.jl <port> [duration_seconds]")
        println("Example: julia collect_and_plot.jl /dev/cu.usbserial-10 10")
        exit(1)
    end

    port_name = ARGS[1]
    duration = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10

    # Collect data
    lines = collect_data(port_name, duration)

    if isempty(lines)
        println("No data collected!")
        exit(1)
    end

    # Parse data
    println("\nParsing data...")
    data = parse_data(lines)

    if isempty(data.timestamps)
        println("No valid data points found!")
        exit(1)
    end

    println("Parsed $(length(data.timestamps)) valid data points")

    # Show balancing time
    balancing_samples = count(s -> s == 1, data.states)
    balancing_time = balancing_samples * 0.01
    println("Time in BALANCING state: $(round(balancing_time, digits=1))s")

    # Compute statistics
    println("\n" * "="^50)
    println("STATISTICS")
    println("="^50)
    stats = compute_statistics(data.pendulum_angles, data.motor_positions)
    println("Target angle:    $(stats["target_angle"])°")
    println("Mean error:      $(round(stats["mean_error"], digits=3))°")
    println("Std deviation:   $(round(stats["std_error"], digits=3))°")
    println("Max error:       $(round(stats["max_error"], digits=3))°")
    println("RMS error:       $(round(stats["rms_error"], digits=3))°")
    println("Motor range:     $(round(stats["motor_range"], digits=1))°")
    println("Duration:        $(round(stats["duration"], digits=1))s")

    # Loop frequency statistics
    if any(f -> f > 0, data.loop_freqs)
        valid_freqs = filter(f -> f > 0, data.loop_freqs)
        println("Loop frequency:  $(round(mean(valid_freqs), digits=0)) Hz (mean), $(round(minimum(valid_freqs), digits=0))-$(round(maximum(valid_freqs), digits=0)) Hz (range)")
    end

    # Overrun count
    if !isempty(data.overruns)
        total_overruns = maximum(data.overruns)  # Cumulative counter
        if total_overruns > 0
            println("Loop overruns:   $(total_overruns) ⚠️")
        else
            println("Loop overruns:   0 ✓")
        end
    end
    println("="^50)

    # Create plots
    println("\nGenerating plots...")
    output_dir = joinpath(dirname(@__FILE__), "experiments")
    plots = create_plots(data, output_dir)

    # Also save raw data as CSV for later analysis
    csv_filename = joinpath(output_dir, "data_capture.csv")
    open(csv_filename, "w") do f
        println(f, "time_s,motor_position_deg,motor_target_deg,pendulum_angle_deg,state,p_term,i_term,d_term,loop_freq_hz,overruns")
        for i in 1:length(data.timestamps)
            println(f, "$(data.timestamps[i]),$(data.motor_positions[i]),$(data.motor_targets[i]),$(data.pendulum_angles[i]),$(data.states[i]),$(data.p_terms[i]),$(data.i_terms[i]),$(data.d_terms[i]),$(data.loop_freqs[i]),$(data.overruns[i])")
        end
    end
    println("Saved raw data: $csv_filename")

    println("\n" * "="^50)
    println("DONE! Generated files:")
    println("="^50)
    for p in plots
        println("  - $p")
    end
    println("  - $csv_filename")
    println("\nClaude Code can now read these plot images for analysis.")
end

main()
