// AS5600 I2C backend: genuine Seeed Studio module.
//
// See docs/BOM.md's "AliExpress / UK (original)" listing. This is the
// original AS5600 library integration (RobTillaart's AS5600 Arduino
// library, over Wire) — this module doesn't need the bus-recovery /
// direct-TWI-polling workarounds the Hailege clone requires (see
// as5600_hailege_clone.h in this directory).
//
// To use this profile: copy this file to ../hw_config.h (gitignored —
// see tools/pi_demo/README.md and flash_if_needed.py, which refuses to
// compile without it).

static void as5600_backend_setup(AS5600 &dev)
{
    Wire.begin();
    Wire.setClock(400000);   // I²C fast mode for short transaction times.
    dev.begin();
}

static void as5600_backend_wait_magnet(AS5600 &dev)
{
    while (!dev.detectMagnet()) { delay(500); }
}

static bool as5600_backend_read(AS5600 &dev, long* out)
{
    *out = dev.rawAngle();
    return true;
}
