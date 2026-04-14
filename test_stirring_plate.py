import time
from stirring_plate import IKADriver

# Change this to match your lab serial port
serial_port = "/dev/ttyACM0"

# Test parameters
test_rpm = 300       # safe low speed for first test
test_temp = 30       # safe low temperature for first test (degrees C)
test_duration = 10   # seconds to run each test


def test_stir(plate):
    print("\n--- STIR TEST ---")
    print(f"Setting RPM to {test_rpm}...")
    plate.setStir(test_rpm)
    plate.startStir()

    time.sleep(2)
    speed = plate.getStirringSpeed()
    print(f"Actual speed reading: {speed} RPM")

    print(f"Stirring for {test_duration} seconds...")
    time.sleep(test_duration)

    plate.stopStir()
    print("Stir stopped.")


def test_heat(plate):
    print("\n--- HEAT TEST (not previously verified) ---")
    print(f"Setting target temp to {test_temp} C...")
    plate.setHeat(test_temp)
    plate.startHeat()

    time.sleep(2)
    temp = plate.getHotplateTemp()
    print(f"Actual temperature reading: {temp} C")

    print(f"Heating for {test_duration} seconds...")
    time.sleep(test_duration)

    plate.stopHeat()
    print("Heat stopped.")


def main():
    print(f"Connecting to plate on {serial_port}...")
    try:
        plate = IKADriver(serial_port)
        print("Connected.")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    test_stir(plate)
    test_heat(plate)

    print("\nAll tests done.")


if __name__ == "__main__":
    main()
