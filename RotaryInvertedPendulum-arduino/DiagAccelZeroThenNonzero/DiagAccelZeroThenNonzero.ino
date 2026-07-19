// Diagnostic: does calling moveByAcceleration(0, true) first (as
// CMD_ENGAGE_MOTOR does) and THEN moveByAcceleration(nonzero, true) a bit
// later (as the first CMD_SET_ACCEL does) break stepping, versus calling
// moveByAcceleration(nonzero, true) directly as DiagFastAccelOnly did
// (which worked)? No I2C, no serial protocol at all — isolates purely
// this call sequence.

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
        Serial.println("FATAL: stepperConnectToPin failed");
        while (true) { ; }
    }
    stepper->setDirectionPin(DIR_PIN);
    stepper->setEnablePin(ENABLE_PIN);
    stepper->setAutoEnable(false);
    stepper->setSpeedInUs(550);
    stepper->setForwardPlanningTimeInMs(8);
    stepper->disableOutputs();

    delay(200);
    Serial.println("enableOutputs()");
    stepper->enableOutputs();

    delay(200);
    Serial.println("moveByAcceleration(0, true)  -- mimics CMD_ENGAGE_MOTOR");
    stepper->moveByAcceleration(0, true);

    delay(300);  // mimics the gap between ENGAGE and the first SET_ACCEL
    Serial.println("moveByAcceleration(5000, true) -- mimics first CMD_SET_ACCEL");
    stepper->moveByAcceleration(5000, true);
}

void loop()
{
    static uint32_t last = 0;
    if (millis() - last > 200)
    {
        last = millis();
        Serial.print("pos="); Serial.print(stepper->getCurrentPosition());
        Serial.print("  speed="); Serial.println(stepper->getCurrentSpeedInMilliHz());
    }
}
