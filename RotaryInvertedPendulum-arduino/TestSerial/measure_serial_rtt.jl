using Statistics
using LibSerialPort

# Function to measure RTT for a single trial with high-resolution timing
function measure_single_rtt(port, test_byte::UInt8; timeout_seconds=1)
    # Flush buffer before measurement
    LibSerialPort.sp_flush(port, LibSerialPort.SP_BUF_BOTH)

    # Use time_ns() for nanosecond resolution instead of now()
    t_start = time_ns()
    LibSerialPort.write(port, [test_byte])

    received_byte = UInt8(0)
    deadline_ns = t_start + timeout_seconds * 1_000_000_000

    # Wait for response with timeout
    while time_ns() < deadline_ns
        if LibSerialPort.bytesavailable(port) > 0
            received_byte = LibSerialPort.read(port, UInt8)
            t_end = time_ns()

            # Convert to milliseconds with microsecond precision
            rtt_ms = (t_end - t_start) / 1_000_000.0

            return received_byte == test_byte ? rtt_ms : nothing
        end
        # Small yield to prevent 100% CPU usage
        # yield()  # Uncomment if needed
    end

    return nothing  # Timeout
end

# Function to measure RTT over multiple trials
function measure_rtt(serial_port::String, baud_rate::Int;
                     trials::Int=100, warmup::Int=5, test_byte::UInt8=0x55)
    println("Testing with baud rate: $baud_rate...")

    # Open the serial port
    port = LibSerialPort.open(serial_port, baud_rate; parity=LibSerialPort.SP_PARITY_NONE)

    # Initialize result variables outside try block for proper scoping
    rtt_times = Float64[]
    timeout_count = 0

    try
        LibSerialPort.set_flow_control(port)
        sleep(1)  # Allow time for port to stabilize
        LibSerialPort.sp_flush(port, LibSerialPort.SP_BUF_BOTH)

        # Warmup trials (discarded)
        println("Running $warmup warmup trials...")
        for i in 1:warmup
            measure_single_rtt(port, test_byte)
        end

        # Actual measurements
        println("Running $trials measurement trials...")

        for i in 1:trials
            rtt = measure_single_rtt(port, test_byte)
            if rtt !== nothing
                push!(rtt_times, rtt)
            else
                timeout_count += 1
                println("Warning: No response received on trial $i")
            end
        end
    finally
        LibSerialPort.close(port)
    end

    # Analyze results
    println("\n" * "="^50)
    println("Results for baud rate $baud_rate:")
    println("="^50)

    if !isempty(rtt_times)
        println("Successful trials: $(length(rtt_times))/$trials")
        println("Timeout count: $timeout_count")
        println("\nRound-Trip Time Statistics:")
        println("  Min:    $(minimum(rtt_times)) ms")
        println("  Max:    $(maximum(rtt_times)) ms")
        println("  Mean:   $(mean(rtt_times)) ms")
        println("  Median: $(median(rtt_times)) ms")
        println("  Std:    $(std(rtt_times)) ms")

        # Percentiles
        if length(rtt_times) >= 10
            p50 = quantile(rtt_times, 0.50)
            p95 = quantile(rtt_times, 0.95)
            p99 = quantile(rtt_times, 0.99)
            println("\nPercentiles:")
            println("  50th:   $p50 ms")
            println("  95th:   $p95 ms")
            println("  99th:   $p99 ms")
        end

        # Calculate effective communication frequency
        avg_rtt = mean(rtt_times)
        max_freq = 1000.0 / avg_rtt  # Hz
        println("\nMax theoretical communication frequency: $(round(max_freq, digits=1)) Hz")
    else
        println("ERROR: No valid RTT measurements!")
        println("Timeout count: $timeout_count")
    end
    println("="^50)
end

# CLI entrypoint
if !isinteractive()
    if length(ARGS) < 2
        println("Usage: julia --project=<PATH_TO_PROJ_ROOT> measure_serial_rtt.jl <serial_port> <baud_rate> [trials]")
        println("Example: julia --project=<PATH_TO_PROJ_ROOT> measure_serial_rtt.jl /dev/cu.usbserial-10 115200 100")
        exit(1)
    end

    serial_port = ARGS[1]

    # Parse baud rate with error handling
    local baud_rate = 115200
    try
        baud_rate = parse(Int, ARGS[2])
    catch
        println("ERROR: Invalid baud rate '$(ARGS[2])'. Must be an integer.")
        exit(1)
    end

    # Parse optional trials argument with error handling
    local trials = 100
    if length(ARGS) >= 3
        try
            trials = parse(Int, ARGS[3])
            if trials <= 0
                println("ERROR: Number of trials must be positive.")
                exit(1)
            end
        catch
            println("ERROR: Invalid trials '$(ARGS[3])'. Must be a positive integer.")
            exit(1)
        end
    end

    measure_rtt(serial_port, baud_rate; trials=trials, warmup=5, test_byte=0x55)
end
