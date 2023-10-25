#!/usr/bin/env python2
#
# PAKET FORMAT  (max 223 bytes)
# +----------------------------------------------------------------------------+
# | SYNC(3) | TYPE(1) | LEN(1) |          PDU(max216)               | CRC16(2) |
# +----------------------------------------------------------------------------+
#                             /                                      \
#                       /                                             \
#                  /              PDU FORMAT (max 216=4*54)            \
#                +-        ---------------------------------------------+
#          0xF0: | PKT_NO(4)=0  | TOTAL_PKTS(4) | FILE_NAME             |
#          0xF1: | PKT_NO(4)    | FILE_DATA(212=53*4)                   |
#          0xF2: | HOWMANY(4)   | ACK_PKT_NO(4) | ACK_PKT_NO(4) .       |
#                +----------------------------------------------     ---+
# 
#
from __future__ import print_function

import struct, time, random, copy
import os, sys
from optparse import OptionParser
import time
import socket, threading

from gnuradio import gr, blocks, digital
from gnuradio import eng_notation
from gnuradio.eng_option import eng_option

import cctrp
from uhd_interface import uhd_transmitter, uhd_receiver

##########################################################################

SYNC = "SY>"
ACK_nums=0
fqueue_tx = []
fqueue_tx_flag = []
fqueue_rx = []
fqueue_rx_flag = []

ack_list_tmp = []


fqueue_ready_to_file = False

##########################################################################
def string_to_hex_list(s):
    #return map(lambda x: hex(ord(x)), s)
    return ' '.join(map(lambda x: hex(ord(x)), s))

##########################################################################
def _get_sock(addr, port, server):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if server:
        sock.bind((addr, port))    
    return sock

################################################################################
# http://www.ip33.com/crc.html
def cal_crc_ccitt16_false(payload):
    crc = 0xFFFF
    poly = 0x1021
    data = list(payload)
    for byte in data:
        crc ^= (ord(byte) << 8)
        for _ in range(8):
            if crc & 0x8000: 
                crc = (crc << 1) ^ poly
            else: 
                crc = crc << 1
    return (crc) & 0xffff

##########################################################################
class thread_tb_tx(threading.Thread):
    def __init__(self, options, t_name="threadFileTransfer_rx"):
        threading.Thread.__init__(self, name = t_name)

        self.sfd = _get_sock("192.168.10.1", options.tx_port, True)
        self.tb  = None

        self.setDaemon(True)
        self.start()

    def run(self):
        while True:
            (data, addr) = self.sfd.recvfrom(1472)
            #print("recvfrom %s %d bytes" %(addr, len(data)))
            if self.tb is not None:
                self.tb.send_pkt(data)
            else:
                pass


################################################################################
class my_top_block(gr.top_block):
  """
  my_top_block
  """
  def __init__(self, options, modulator_class, demodulator_class):
    gr.top_block.__init__(self, "my_top_block")
    
    ##################################################################
    # Work-around to get the modulation's bits_per_symbol
    kargs = modulator_class.extract_kwargs_from_options(options)
    #symbol_rate = options.bitrate / modulator(**args).bits_per_symbol()
    self.sps = modulator_class(**kargs).samples_per_symbol()
    print("### modulation: %s self.sps = %d" % (options.modulation, self.sps))
    print("### packet duaration ~%.05fs" % ((16+4+3+255*2+2+8)*8*self.sps*1.0/options.tx_bandwidth))
    self.g_interp = self.sps * 8           # bytes --> bits

    ##################################################################
    # Rx
    if options.rx_infile is not None:
        self.f = blocks.file_source(gr.sizeof_gr_complex, options.rx_infile, options.repeat)
        self.src = blocks.throttle(gr.sizeof_gr_complex, options.rx_bandwidth*options.speed)
        self.connect(self.f, self.src)
    elif options.rx_freq is not None:
        self.src = uhd_receiver(options.rx_args, options.rx_bandwidth, options.rx_freq, 0, options.rx_gain,
             options.rx_spec, options.rx_antenna, options.clock_source, options.verbose)
    else:
        raise Exception("Error: -f or --rx-infile must be specified !!!")

    kargs = demodulator_class.extract_kwargs_from_options(options)
    self.demodulator = demodulator_class(**kargs)

    access_code = "10010011000010110101000111011110"
    threshold   =  3
    packet_len  =  3+(255+1)*2       # golay24(3)+225(RS(223,32))*2(viterbi 1/2)
    self.sync2pdu = cctrp.sync_to_pdu_packed(packet_len, access_code, threshold)

    self.spbcp_decoder = cctrp.spbcp_decode(1, 1, 1, True);
    self.msg_debug = blocks.message_debug()
    self.msg_sink  = blocks.socket_pdu("UDP_CLIENT", "192.168.10.1", str(options.rx_port))
    # connect
    self.connect(self.src, self.demodulator, self.sync2pdu)
    self.msg_connect((self.sync2pdu, 'out'), (self.spbcp_decoder, 'in'))
    #self.msg_connect((self.spbcp_decoder, 'out'), (self.msg_debug, 'print_pdu'))
    self.msg_connect((self.spbcp_decoder, 'out'), (self.msg_sink, 'pdus'))

    ##################################################################
    #Tx
    self.msgs = blocks.message_burst_source(gr.sizeof_char, 64)

    access_code = "10010011000010110101000111011110"
    self.spbcp_encoder = cctrp.spbcp_encode(access_code, 1, 1, 1, 16, 8, False);

    kargs = modulator_class.extract_kwargs_from_options(options)
    self.modulator = modulator_class(**kargs)

    self.s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, self.g_interp)
    self.tag_align = cctrp.burst_stream_tag_align(self.g_interp, False)

    self.amp = blocks.multiply_const_cc(options.tx_amp)

    if options.tx_outfile is not None:
        self.sink =  blocks.file_sink(gr.sizeof_gr_complex, options.tx_outfile)
    elif options.tx_freq is not None:
        self.sink = uhd_transmitter(options.tx_args, options.tx_bandwidth, options.tx_freq, options.tx_looffset, options.tx_gain,
             options.tx_spec, options.tx_antenna, options.clock_source, options.verbose)
    else:
        print("--freq or --tx-outfile must be specified, otherwise the stream will be sent to null sink !!!")
        self.sink = blocks.null_sink(gr.sizeof_gr_complex)

    self.tagdgb = blocks.tag_debug(gr.sizeof_gr_complex, "TAGDEBUG")

    # connect
    self.connect(self.msgs, self.spbcp_encoder, self.modulator)
    #self.connect(self.modulator, self.amp, self.sink)
    self.connect(self.modulator, self.s2v, (self.tag_align, 0), self.amp, self.sink)
    self.connect((self.spbcp_encoder, 1), (self.tag_align,1))
    #self.connect(self.amp, self.tagdgb)

  def send_pkt(self, pkt, eof=False):
    if eof:
      msg = gr.message(1)  # tell self._pkt_input we're not sending any more packets
    else:
      msg = gr.message_from_string(pkt)
    msgq = self.msgs.msgq()
    msgq.insert_tail(msg)
    #print("sending %d bytes in tb" % (len(pkt)))

  @staticmethod
  def add_options(normal, expert, debug):
    uhd_transmitter.add_options(normal)
    expert.add_option("", "--tx-bandwidth", type="eng_float", default=1e6,
                help = "set sample rate/bandwidth [default=%default]")
    expert.add_option("", "--bit-rate", type="eng_float", default=500e3,
                help = "set baseband bit rate [default=%default]")
    expert.add_option("", "--tx-amp", type="eng_float", default=0.8,
                help = "set tx amplitude [0<x<1.0] [default=%default]")
    debug.add_option("","--tx-outfile", default=None,
                      help="Output file for modulated samples")
    
    uhd_receiver.add_options(normal)
    expert.add_option("", "--rx-bandwidth", type="eng_float", default=1e6,
                help = "set sample rate/bandwidth [default=%default]")
    expert.add_option("", "--bit-rate", type="eng_float", default=500e3,
                help = "set baseband bit rate [default=%default]")
    debug.add_option("", "--rx-infile", type="string", default=None,
                help = "select rx iq data from file [default=%default]")
    debug.add_option("", "--repeat", action="store_true", default=False,
                help = "set repeat when using file as the data source [default=%default]")
    debug.add_option("", "--speed", type="float", default=1.0,
                help = "set reading speed when using infile. 1x... [default=%default]")

##########################################################################
def gen_packet(msg, pdutype, pkt_len=223):
    apdu = []

    global SYNC
    if pdutype == 'HEAD':
        TYPE = struct.pack('>B', 0xF0)
    elif pdutype == 'DATA':
        TYPE = struct.pack('>B', 0xF1)
    elif pdutype == 'ACK':
        TYPE = struct.pack('>B', 0xF2)
    else:
        raise Exception('Unknown type...')

    overhead = len(SYNC) + 1 + len(TYPE) + 2
    payload_len = pkt_len - overhead

    if len(msg) > (payload_len):
        msg = msg[0 : payload_len]
        padding = []
    else:
        padding = struct.pack('!B', 0x55) * (pkt_len-len(msg)-overhead)
   
    LEN = struct.pack('!B', len(msg)) # big-endian unsigned short
    
    apdu = SYNC + TYPE + LEN + msg
    apdu += ''.join(padding)
    apdu += struct.pack('!H', cal_crc_ccitt16_false(apdu) & 0xffff)

    #print ">>>>%d [%s]" % (len(apdu), string_to_hex_list(apdu))

    return apdu


##########################################################################
class file_transfer_udp_tx(threading.Thread):
    def __init__(self, options, t_name="threadFileTransfer_tx"):
        threading.Thread.__init__(self, name = t_name)

        self.sfd = _get_sock(options.tx_ip, options.tx_port, False)
        self.tx_ip = options.tx_ip
        self.tx_port = options.tx_port
        self.tx_file = options.tx_file
        self.fd     = None

        self.pkts_num_rest = 20
        self.pkts_num_send = 0
        
        self.pktsize = options.pktsize
        self.pkt_data_len = self.pktsize - 7 - 4

        self.pkt_no = 0
        self.pkt_total = 0
        self.pdu = []

        self.tx_queue_idx = 0
        self.rx_ack_idx = 0

        global fqueue_tx, fqueue_tx_flag
        fqueue_tx = []
        fqueue_tx_flag = []
        if self.tx_file is not None:
            self.fd = open(self.tx_file, 'rb')
            try:
                while True:
                    if self.pkt_no == 0:
                        fqueue_tx = []
                        fqueue_tx.append('')
                        fqueue_tx_flag.append(0);
                        print("pkt_no->%d" % (self.pkt_no))
                    
                    s = self.fd.read(self.pkt_data_len)  # type, pkt_no
                    if not len(s):
                        break
                    #print "%d, %d" %(pkt_no, len(s))
                    self.pkt_no += 1

                    self.pdu = struct.pack('!L', self.pkt_no) + s

                    msg = gen_packet(self.pdu, 'DATA', pkt_len=self.pktsize)
                    #print("pkt_no->%4d len(pdu)->%d len(msg)->%d" % (self.pkt_no, len(self.pdu), len(msg)))
                    fqueue_tx.append(msg)
                    fqueue_tx_flag.append(0);
                #update the 0 pkt
                (path, filename) = os.path.split(options.tx_file)
                self.pdu = struct.pack('!L', 0) + struct.pack('!L', self.pkt_no) + "r-" + filename
                msg = gen_packet(self.pdu, 'HEAD', pkt_len = self.pktsize)
                print("pkt_no->%4d len(pdu)->%d len(msg)->%d  total_pkts=%d" % (0, len(self.pdu), len(msg), len(fqueue_tx)))
                fqueue_tx[0] = msg
                fqueue_tx_flag[0] = 0
            finally:
                self.fd.close()
        
        self.setDaemon(True)
        self.start()

    def send_pkt(self, pkt):
        self.sfd.sendto(pkt, (self.tx_ip, self.tx_port))
        #print("sending %d bytes to %s:%s" % (len(pkt), self.tx_ip, self.tx_port))

    def run(self):
        #global fqueue_rx, fqueue_rx_flag, fqueue_tx, fqueue_tx_flag
        while True:
            # tx queue
            if (len(fqueue_tx_flag) > 0) and (self.pkts_num_send < self.pkts_num_rest):
                # need to send head pkt first
                if fqueue_tx_flag[0] == 0:      # head pkt not acked
                    self.send_pkt(fqueue_tx[0])
                    self.tx_queue_idx = 1
                    self.pkts_num_send += 1
                    time.sleep(0.02)
                else:                           # send data pkt
                    if(self.tx_queue_idx > (len(fqueue_tx_flag)-1)):
                        self.tx_queue_idx = 1
                        self.pkts_num_send = self.pkts_num_rest   # force to send ack after 1 cycle
                    if fqueue_tx_flag[self.tx_queue_idx] == 0:     # not acked pkt
                        self.send_pkt(fqueue_tx[self.tx_queue_idx])
                        self.pkts_num_send += 1
                        time.sleep(0.005)
                    self.tx_queue_idx += 1
                if fqueue_tx_flag.count(1) == len(fqueue_tx_flag):
                    print("sending complete...")
                    exit(0)
                    break;
            # rx queue flag / only send the correct pkt ack when received correctly
            else:
                global ack_list_tmp
                self.pkts_num_send = 0
                pdu = ''
                if len(ack_list_tmp) > 0:
                    i = 0
                    while i < 40:
                        try:
                            ack_pkt = ack_list_tmp.pop(0)
                            pdu += struct.pack('>L', ack_pkt)
                            i += 1
                        except Exception as e:          # empty
                             break
                    if i > 0:
                        global ACK_nums
                        pdu = struct.pack('>L', i) + pdu
                        pkt = gen_packet(pdu, 'ACK', pkt_len=self.pktsize)
                        ACK_nums=ACK_nums+1
                        self.send_pkt(pkt)
                        #print("sending ack pkt...")
                        time.sleep(0.02)
                else:
                    time.sleep(0.02)


    def add_options(normal, expert, debug):
        normal.add_option("", "--tx-ip", type="string", default="127.0.0.1",
                        help="set the port of ip destinatiom, [default=%default]")
        normal.add_option("-p", "--tx-port", type="int", default=58801,
                        help="set the port of udp destinatiom, [default=%default]")
        normal.add_option("-f", "--tx-file", type="string", default=None,
                        help="set the port of file to be send, [default=%default]")
        normal.add_option("", "--pktsize", type="int", default=223,
                        help="set pktsize to be send, [default=%default]")
    add_options = staticmethod(add_options)


##########################################################################
class file_transfer_udp_rx(threading.Thread):
    def __init__(self, options, t_name="threadFileTransfer_rx"):
        threading.Thread.__init__(self, name = t_name)
        self.options = options
        self.sfd = _get_sock("0.0.0.0", options.rx_port, True)
        self.tx_ip = options.tx_ip
        self.tx_port = options.tx_port
        self.setDaemon(True)
        self.start()

    def send_pkt(self, pkt):
        self.sfd.sendto(pkt, (self.tx_ip, self.tx_port))

    def parse_pkt(self, pkt):
        global fqueue_rx, fqueue_rx_flag
        global ack_list_tmp
        if pkt[0:3] != SYNC:
            return
        pkt_type    = struct.unpack('!B', pkt[3:4])[0]
        pkt_pdu_len = struct.unpack('!B', pkt[4:5])[0]
        crc_recv    = struct.unpack('>H', pkt[-2:])[0]
        crc_calc    = cal_crc_ccitt16_false(pkt[0:-2]) & 0xffff
        #print("type: %02X, len=%d, crc: %02X==%02X" % (pkt_type, pkt_pdu_len, crc_recv, crc_calc))
        if crc_recv != crc_calc:
            return
        if pkt_pdu_len > (len(pkt)-6):
            return
        
        if pkt_type == 0xF0:
            total_pkts = struct.unpack('>L', pkt[9:13])[0]
            filename   = pkt[13:13+(pkt_pdu_len-8)]
            #print("total_pkts:%d filename %s" % (total_pkts, filename))
            fqueue_rx = [''] * (total_pkts+1)
            fqueue_rx_flag = [0] * (total_pkts+1)
            fqueue_rx[0]       = filename
            fqueue_rx_flag[0]  = 1
            ack_list_tmp = [0]                 # send head ack only
        elif pkt_type == 0xF1:
            pkt_no = struct.unpack('>L', pkt[5:9])[0]
            pkt_data = pkt[9:9+(pkt_pdu_len-4)]
            if len(fqueue_rx) > 0:
                fqueue_rx[pkt_no]       = pkt_data
                fqueue_rx_flag[pkt_no]  = 1
                ack_list_tmp.append(pkt_no)
        elif pkt_type == 0xF2:
            ack_pkt_num = struct.unpack('>L', pkt[5:9])[0]
            pdu_idx = 0
            while pdu_idx < (pkt_pdu_len-4):
                ack_no = struct.unpack('>L', pkt[9+pdu_idx:9+pdu_idx+4])[0]
                pdu_idx += 4
                fqueue_tx_flag[ack_no] = 1 # need to assure that not exeed the boudary

    def run(self):
        while True:
            (data, addr) = self.sfd.recvfrom(1472)
            #print("recvfrom %s %d bytes" %(addr, len(data)))
            if self.options.sim:
                probability = random.random()
                if probability < 0.2:
                    #print("dropped...")
                    continue

            self.parse_pkt(data)


    def add_options(normal, expert, debug):
        normal.add_option("", "--rx-port", type="int", default=58802,
                        help="set the port of udp destinatiom, [default=%default]")
        debug.add_option("", "--sim", action="store_true", default=False,
                        help = "simulate the channel PER [default=%default]")
    add_options = staticmethod(add_options)


##########################################################################
def main():
    mods = cctrp.modulation_utils.type_1_mods()
    demods = cctrp.modulation_utils.type_1_demods()

    description = "test_cctrp_file_trx by CHEN,Jian"
    parser = OptionParser(option_class=eng_option, conflict_handler="resolve", description=description)
    expert_grp = parser.add_option_group("Expert")
    debug_grp = parser.add_option_group("Debug")
    parser.add_option("-m", "--modulation", type="choice", choices=mods.keys(),
                      default='gmsk',
                      help="Select modulation from: %s [default=%%default]"
                            % (', '.join(mods.keys()),))

    file_transfer_udp_tx.add_options(parser, expert_grp, debug_grp)
    file_transfer_udp_rx.add_options(parser, expert_grp, debug_grp)
    my_top_block.add_options(parser, expert_grp, debug_grp)

    for mod in mods.values():
        mod.add_options(expert_grp)

    for demod in demods.values():
        demod.add_options(expert_grp)

    (options, args) = parser.parse_args()

    thread_tbtx  = thread_tb_tx(options)
    tb   = my_top_block(options, mods[options.modulation], demods[options.modulation])
    thread_tbtx.tb = tb

    r = gr.enable_realtime_scheduling()
    if r != gr.RT_OK:
        print("Warning: failed to enable realtime scheduling")

    tb.start()

    thread_ftx = file_transfer_udp_tx(options)
    thread_frx = file_transfer_udp_rx(options)
    
    begin_time=time.time()
    file_rx_complete = 0
    mark_times=0
    while True:
        time.sleep(0.5)
        if len(fqueue_tx_flag) > 0:
            tx_percent = fqueue_tx_flag.count(1) * 100.0 / len(fqueue_tx_flag)
        else:
            tx_percent = 0
        if len(fqueue_rx_flag) > 0:
            rx_percent = fqueue_rx_flag.count(1) * 100.0 / len(fqueue_rx_flag)
        else:
            rx_percent = 0

        if( rx_percent==100 and mark_times==0):
            end_time=time.time()
            cost_time=end_time-begin_time
            ostr = "\rTransfering  Tx:%2.02f%%, Rx:%2.02f%% , this process costs you %2.05f s" % (tx_percent, rx_percent,cost_time)
            print(ostr)
            print('the number of ack is', ACK_nums)
            mark_times=1
        else:
            ostr = "\rTransfering  Tx:%2.02f%%, Rx:%2.02f%%" % (tx_percent, rx_percent)
        sys.stdout.write(ostr)
        sys.stdout.flush()
        
        if (len(fqueue_rx_flag) > 0):
            if (file_rx_complete==0) and (fqueue_rx_flag.count(1) == len(fqueue_rx_flag)):
                file_rx_complete = 1
                fd = None
                for i in range(len(fqueue_rx_flag)):
                    if i == 0:
                        fd = open(fqueue_rx[0], 'wb')
                    else:
                        fd.write(fqueue_rx[i])
                fd.close()
        if(rx_percent==100):
            tb.stop()
            quit()   

if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

