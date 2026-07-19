using Dates
using LibSerialPort
using Printf
using Statistics

# Define the serial port and baud rate
serial_port = "/dev/cu.usbserial-110"  # Replace with your serial port
baud_rate = 1000000

# Function to check if the Arduino is ready with retry mechanism
function check_arduino_ready(port; retries=3, timeout_seconds=1)
    attempts = 0
    while attempts < retries
        println("Checking if Arduino is ready... (Attempt $(attempts + 1)/$retries)")  # Debug: show attempt number
        LibSerialPort.write(port, UInt8('R'))  # Send 'R' for ready check

        # Flush the serial buffer
        LibSerialPort.flush(port, LibSerialPort.SP_BUF_BOTH)

        # Wait for the response (1 byte for ready acknowledgment)
        response = UInt8[]
        t_timeout = now() + Second(timeout_seconds)

        while now() < t_timeout
            if LibSerialPort.bytesavailable(port) > 0
                byte = LibSerialPort.read(port, UInt8)
                println("Received byte: $(byte)")  # Debug: show each byte received
                push!(response, byte)
            end
        end

        # Check if we received the expected response ('R')
        if length(response) == 1 && response[1] == UInt8('R')
            println("Arduino is ready!")
            return true
        else
            println("Error: Arduino is not responding correctly. Retrying...")
            attempts += 1
            sleep(0.5)  # Small delay before retrying
        end
    end

    println("Failed to get a valid response after $retries attempts.")
    return false
end

# Function to query the Arduino for sine or cosine value
function query_wave_value(port, request_char::Char)
    # Send the request to the Arduino
    # println("Sending request: $request_char")
    message = [UInt8(request_char)]
    LibSerialPort.sp_blocking_write(port.ref, message, 100)  # Blocking write with a timeout of 100ms

    # Prepare to receive 2 bytes
    response = zeros(UInt8, 2)  # Preallocate a buffer for 2 bytes
    nbytes_read = LibSerialPort.sp_blocking_read(port.ref, Ref(response, 1), 2, 500)  # Blocking read with a 500ms timeout

    # Check if we received exactly 2 bytes
    if nbytes_read == 2
        # println("Received full response: $response")
        float_value = decode_wave_value(response)
        return float_value
    else
        println("Error: Expected 2 bytes, but received $nbytes_read bytes")
        return nothing
    end
end

function decode_wave_value(bytes::Vector{UInt8})
    # Combine the two bytes into a signed 16-bit integer
    int_value = reinterpret(Int16, UInt8[bytes[1], bytes[2]])[1]
    # Scale back to float in the range [-1.0, 1.0]
    return int_value / 32767.0
end

# Main function to interact with the Arduino
function main()
    # Open the serial port
    port = LibSerialPort.open(serial_port, baud_rate; parity=LibSerialPort.SP_PARITY_NONE)
    LibSerialPort.set_flow_control(port)

    sleep(1) # Allow time for the Arduino to initialize

    # Check if the Arduino is ready with retry mechanism
    if check_arduino_ready(port)
        # Track the start time of the entire loop
        start_time = now()
        loop_duration = Second(5)  # Run the loop for 5 seconds
        last_frequency_time = now()

        # List to store iteration durations for moving average
        iteration_times = Float64[]

        # Start the loop for querying the sine and cosine values
        while now() - start_time < loop_duration
            iteration_start_time = now()  # Start time of the current iteration

            # Query the Arduino for sine and cosine values
            sine_value = query_wave_value(port, 'S')
            cosine_value = query_wave_value(port, 'C')

            # if sine_value !== nothing
            #     @printf("Sine Value: %.4f\n", sine_value)
            # end
            # if cosine_value !== nothing
            #     @printf("Cosine Value: %.4f\n", cosine_value)
            # end

            # Track the duration of this iteration
            iteration_duration = (now() - iteration_start_time).value / 1000  # Convert to seconds
            push!(iteration_times, iteration_duration)

            # # Print the iteration duration
            # @printf("Iteration duration: %.4f seconds\n", iteration_duration)

            # Every 0.5 seconds, print the moving average loop frequency
            if now() - last_frequency_time >= Millisecond(500)
                if !isempty(iteration_times)
                    avg_freq = 1 / mean(iteration_times)
                    @printf("Moving average loop frequency: %.2f Hz\n", avg_freq)
                end
                last_frequency_time = now()  # Update the last frequency print time

                # Clear the iteration times list for the next period
                empty!(iteration_times)
            end
        end
    else
        println("Error: Arduino not ready. Exiting.")
    end

    # Close the serial port
    LibSerialPort.close(port)
end

# Only run the main function if the script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
