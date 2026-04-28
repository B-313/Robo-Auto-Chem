import time
import serial
import re

#############################################
# Custom class definition for the IKA plate #
#############################################

class IKADriver:
    serialCom = serial.Serial()

    def __init__(self, serial_port):
        global serialCom
        serialCom = serial.Serial(
            port=serial_port,
            baudrate=9600,
            parity=serial.PARITY_NONE,     # Changed this because apparently this model uses this serial
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,     # This model actually uses this
            timeout=1
        )
        time.sleep(1)  # It needs time to rest

    def send(self, cmd):
        global serialCom
        serialCom.reset_input_buffer()
        serialCom.write((cmd + "\r\n").encode())
        time.sleep(0.1)
        return serialCom.read_until(b"\n")

    # STIR CONTROL

    def startStir(self):
        self.send("START_4")

    def stopStir(self):
        self.send("STOP_4")

    def setStir(self, stir):
        self.send(f"OUT_SP_4 {stir}")

    def startMaxStir(self):
        # Typical max is 1500 RPM (safe default)
        self.setStir(1500)
        time.sleep(0.1)
        self.startStir()

    def run_stir_session(self, rpm, duration_seconds, temp=None):
        # Start stirring and optional heating, wait, then stop both safely.
        try:
            self.setStir(rpm)
            self.startStir()
            if temp is not None:
                self.setHeat(temp)
                self.startHeat()
            time.sleep(duration_seconds)
        finally:
            self.stopStir()
            if temp is not None:
                self.stopHeat()

    # HEAT CONTROL # not tried yet

    def startHeat(self):
        self.send("START_1")

    def stopHeat(self):
        self.send("STOP_1")

    def setHeat(self, temp):
        self.send(f"OUT_SP_1 {temp}")

    # READ VALUES

    def getStirringSpeed(self):
        response = self.send("IN_PV_4")
        print("RAW:", response)  # debug
        stringx = response.decode('ascii')
        s = re.findall(r"[-+]?\d*\.?\d+", stringx)
        return s[0] if s else None

    def getHotplateTemp(self):
        response = self.send("IN_PV_2")
        stringx = response.decode('ascii')
        s = re.findall(r"[-+]?\d*\.?\d+", stringx)
        return s[0] if s else None
    
"""
# Sample usage:
# It's been tested, it works.

driver = IKADriver('/dev/ttyACM0')

# Start max-speed stirring
driver.startMaxStir()

time.sleep(10)

# Check speed
print("Speed:", driver.getStirringSpeed())

# Stop stirring
driver.stopStir()

"""
