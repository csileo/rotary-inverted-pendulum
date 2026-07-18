// Minimal echo sketch at 2 Mbaud.
// Lights LED on first received byte to confirm serial works.
void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    Serial.begin(2000000);
    digitalWrite(LED_BUILTIN, HIGH);
}

void loop() {
    if (Serial.available() > 0) {
        uint8_t b = Serial.read();
        Serial.write(b);  // echo
        digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));  // toggle on each byte
    }
}
