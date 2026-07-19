// Diagnostic: does FastAccelStepper alone (no I2C, no serial protocol, no
// watchdog) actually spin the motor on this rig? Isolates whether a stall
// is specific to LowLevelServer's I2C/serial/watchdog interactions, or a
// more basic FastAccelStepper/Timer1 issue on this hardware.
//
// No host commands needed: engages and commands a fixed accel ~0.5 s after
// boot. Watch the motor and the "pos=" printout (should climb steadily).

#include <FastAccelStepper.h>

#define DIR_PIN 2
#define STEP_PIN 9     // Timer1 OC1A
#define ENABLE_PIN 5

FastAccelStepperEngine engine = FastAccelStepperEngine();
FastAccelStepper *stepper = NULL;

void setup()
{
    Serial.begin(115200);
    while (!Serial) { ; }

    engine.init();
    stepper = engine.stepperConnectToPin(STEP_PIN);
    if (!stepper)
    {
        Serial.println("FATAL: stepperConnectToPin failed (wrong pin for Timer1 OC1A/OC1B?)");
        while (true) { ; }
    }
    stepper->setDirectionPin(DIR_PIN);
    stepper->setEnablePin(ENABLE_PIN);
    stepper->setAutoEnable(false);

    int8_t rc_speed = stepper->setSpeedInUs(550);  // matches LowLevelServer's cap (~7 rad/s)
    Serial.print("setSpeedInUs rc=");
    Serial.println(rc_speed);
    stepper->setForwardPlanningTimeInMs(8);
    stepper->disableOutputs();

    delay(500);
    Serial.println("Enabling outputs...");
    stepper->enableOutputs();
    delay(200);

    Serial.println("Commanding accel = 5000 steps/s^2 (~19.6 rad/s^2), allow_reverse=true");
    stepper->moveByAcceleration(5000, true);
}

void loop()
{
    static uint32_t last = 0;
    if (millis() - last > 200)
    {
        last = millis();
        Serial.print("pos=");
        Serial.print(stepper->getCurrentPosition());
        Serial.print("  speed=");
        Serial.println(stepper->getCurrentSpeedInMilliHz());
    }
}
