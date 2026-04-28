from stirring_plate import IKADriver


def connect_plate(port):
    # Connect to the IKA stirring plate. Returns None if unavailable.
    try:
        plate = IKADriver(port)
        print(f"Stirring plate connected on {port}")
        return plate
    except Exception as e:
        print(f"Stirring plate init failed: {e}")
        return None


def run_stir_session(plate, rpm, duration_seconds, temp=None):
    # Start stirring (and optional heating), block for duration, then stop safely.
    # plate: IKADriver instance or None (no-op if None)
    if plate is None:
        return
    plate.run_stir_session(rpm, duration_seconds, temp=temp)
