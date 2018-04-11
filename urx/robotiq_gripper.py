import crcmod
import serial
import binascii
import time


class RobotiqGripper(object):
    def __init__(self, port='/dev/ttyUSB0'):
        self.ser = serial.Serial(port=port,  # ensure port writable: sudo chmod 777 /dev/ttyUSB0
                                 baudrate=115200,
                                 timeout=1,
                                 parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE,
                                 bytesize=serial.EIGHTBITS)
        self.crc16 = crcmod.mkCrcFun(0x18005, rev=True, initCrc=0xFFFF, xorOut=0x0000)
        # Robot Input / Status
        self.ACTION_REQUEST = 0x00  # byte0
        self.POSITION_REQUEST = 0x00  # byte3
        self.SPEED = 0x00  # byte4
        self.FORCE = 0x00  # byte5
        # Robot Output / Functionalities
        self.GRIPPER_STATUS = 0x00  # byte0
        self.FAULT_STATUS = 0x00  # byte2
        self.POS_REQUEST_INFO = 0x00  # byte3
        self.POSITION = 0x00  # byte4
        self.CURRENT = 0x00  # byte5

    def activation_request(self):
        self.ACTION_REQUEST = 0x00
        self.ser.write(self.get_control_command())  # clear rAct
        time.sleep(0.7)
        raw_data = self.ser.readline()
        _ = binascii.hexlify(raw_data)

        self.ACTION_REQUEST = 0x01
        self.ser.write(self.get_control_command())  # set rAct
        time.sleep(1.7)
        raw_data = self.ser.readline()
        _ = binascii.hexlify(raw_data)

        self.ACTION_REQUEST = 0x09
        self.ser.write(self.get_control_command())  # set rAct
        time.sleep(0.7)
        raw_data = self.ser.readline()
        _ = binascii.hexlify(raw_data)

        self.ser.write(self.get_reading_command(1))  # read Gripper status until the activation is completed
        time.sleep(0.7)
        raw_data = self.ser.readline()
        _ = binascii.hexlify(raw_data)
        print 'gripper activated.'

    def set_speed(self, speed=0xFF):
        self.SPEED = speed
        self.ser.write(self.get_control_command())  # clear rAct
        time.sleep(0.7)
        raw_data = self.ser.readline()
        _ = binascii.hexlify(raw_data)

    def set_force(self, force=0xFF):
        self.FORCE = force
        self.ser.write(self.get_control_command())  # clear rAct
        time.sleep(0.7)
        raw_data = self.ser.readline()
        _ = binascii.hexlify(raw_data)

    def get_gripper_pos(self):
        self.ser.write(self.get_reading_command())
        raw_data = self.ser.readline()
        data = binascii.hexlify(raw_data)  # hexstr
        self.parsing_reading_sequence(data)
        return self.POSITION

    def get_object_detection_status(self):
        self.ser.write(self.get_reading_command())
        raw_data = self.ser.readline()
        data = binascii.hexlify(raw_data)  # hexstr
        self.parsing_reading_sequence(data)
        gOBJ = (self.GRIPPER_STATUS & 0xC0) >> 6  # object detection status
        if gOBJ == 0:
            print 'Fingers are in motion towards requested position. No object detected.'
        elif gOBJ == 1:
            print 'Fingers have stopped due to a contact while opening before requested position. ' \
                  'Object detected opening.'
        elif gOBJ == 2:
            print 'Fingers have stopped due to a contact while closing before requested position. ' \
                  'Object detected closing.'
        elif gOBJ == 3:
            print 'Fingers are at requested position. No object detected or object has been loss / dropped.'
        return gOBJ

    def gripper_close(self):
        self.POSITION_REQUEST = 0xFF
        self.ser.write(self.get_control_command())
        time.sleep(0.7)
        raw_data = self.ser.readline()
        _ = binascii.hexlify(raw_data)

    def gripper_open(self):
        self.POSITION_REQUEST = 0x00
        self.ser.write(self.get_control_command())
        raw_data = self.ser.readline()
        _ = binascii.hexlify(raw_data)
        time.sleep(0.7)

    def set_gripper_pos(self, pos):
        self.POSITION_REQUEST = pos
        self.ser.write(self.get_control_command())
        raw_data = self.ser.readline()
        _ = binascii.hexlify(raw_data)
        time.sleep(0.7)

    def parsing_reading_sequence(self, sequence):
        slave_id = sequence[0:2]
        func_code = sequence[2:4]
        num_data_byte = self.str2byte(sequence[4:6])
        num_reg = num_data_byte / 2
        if num_reg == 1:
            self.GRIPPER_STATUS = self.str2byte(sequence[6:8])
        elif num_reg == 2:
            self.GRIPPER_STATUS = self.str2byte(sequence[6:8])
            self.FAULT_STATUS = self.str2byte(sequence[10:12])
            self.POS_REQUEST_INFO = self.str2byte(sequence[12:14])
        elif num_reg == 3:
            self.GRIPPER_STATUS = self.str2byte(sequence[6:8])
            self.FAULT_STATUS = self.str2byte(sequence[10:12])
            self.POS_REQUEST_INFO = self.str2byte(sequence[12:14])
            self.POSITION = self.str2byte(sequence[14:16])
            self.CURRENT = self.str2byte(sequence[16:18])

    def parsing_control_sequence(self, sequence):
        slave_id = sequence[0:2]
        func_code = sequence[2:4]
        first_reg_addr = sequence[4:8]
        num_written_reg = sequence[8:12]
        crc_calc = sequence[12:16]

    def get_control_command(self):
        slave_id = '09'
        func_code = '10'  # Function Code 16 (Preset Multiple Registers)
        first_reg_addr = '03E8'  # Address of the first register
        num_reg = '0003'
        num_data_byte = '06'
        action_request = self.byte2str(self.ACTION_REQUEST) + '00'
        position_request = '00' + self.byte2str(self.POSITION_REQUEST)
        speed = self.byte2str(self.SPEED)
        force = self.byte2str(self.FORCE)
        control_command = slave_id + func_code + first_reg_addr + num_reg + \
                          num_data_byte + action_request + position_request + speed + force
        crc_calc = self.crc16(control_command.decode('hex'))
        control_command += self.byte2str(crc_calc & 0x00FF)
        control_command += self.byte2str(crc_calc >> 8)
        return control_command.decode('hex')

    def get_reading_command(self, num_reg_to_read=3):
        slave_id = '09'
        func_code = '03'  # Function Code 03 (Read Holding Registers)
        first_reg_addr = '07D0'  # Address of the first register
        num_reg = '000{}'.format(num_reg_to_read)  # Number of registers requested (3)
        reading_command = slave_id + func_code + first_reg_addr + num_reg
        crc_calc = self.crc16(reading_command.decode('hex'))
        reading_command += self.byte2str(crc_calc & 0x00FF)
        reading_command += self.byte2str(crc_calc >> 8)
        return reading_command.decode('hex')

    @staticmethod
    def str2byte(string):
        s2b_dict = {'0': 0, '1': 1, '2': 2, '3': 3,
                    '4': 4, '5': 5, '6': 6, '7': 7,
                    '8': 8, '9': 9, 'A': 10, 'B': 11,
                    'C': 12, 'D': 13, 'E': 14, 'F': 15,
                    'a': 10, 'b': 11, 'c': 12, 'd': 13,
                    'e': 14, 'f': 15}
        byte = s2b_dict[string[0]] * 16 + s2b_dict[string[1]]
        return byte

    @staticmethod
    def byte2str(byte):
        b2s_dict = {0: '0', 1: '1', 2: '2', 3: '3',
                    4: '4', 5: '5', 6: '6', 7: '7',
                    8: '8', 9: '9', 10: 'A', 11: 'B',
                    12: 'C', 13: 'D', 14: 'E', 15: 'F'}
        string = ''
        string += b2s_dict[byte / 16]
        string += b2s_dict[byte % 16]
        return string
