#! /usr/bin/env python3

# Copyright 2018 Intel Corporation.
# The source code, information and material ("Material") contained herein is
# owned by Intel Corporation or its suppliers or licensors, and title to such
# Material remains with Intel Corporation or its suppliers or licensors.
# The Material contains proprietary information of Intel or its suppliers and
# licensors. The Material is protected by worldwide copyright laws and treaty
# provisions.
# No part of the Material may be used, copied, reproduced, modified, published,
# uploaded, posted, transmitted, distributed or disclosed in any way without
# Intel's prior express written permission. No license under any patent,
# copyright or other intellectual property rights in the Material is granted to
# or conferred upon you, either expressly, by implication, inducement, estoppel
# or otherwise.
# Any license under such intellectual property rights must be express and
# approved by Intel in writing.

import os
import sys
import argparse
import numpy as np

#if sys.version_info[0] != 3:
#    sys.stdout.write("Attempting to run with a version of Python != 3.x\n")
#   sys.exit(1)

from Controllers.EnumController import *
from Controllers.FileIO import *
from Models.Blob import *
from Models.EnumDeclarations import *
from Models.MyriadParam import *
from Views.Summary import *
from Views.Graphs import *
import Controllers.Globals as GLOBALS
from Controllers.Scheduler import load_myriad_config, load_network
from Controllers.PingPong import ppInit

major_version = np.uint32(2)
release_number = np.uint32(0)


def parse_args():
    parser = argparse.ArgumentParser(description="mvNCProfile.py profiles a Caffe or Tensorflow network on the Movidius Neural Computer Stick\n")
    parser.add_argument('-network', type=str, help='Network file (.prototxt, .meta, .pb, .protobuf)')
    parser.add_argument('-w', dest='weights', type=str, help='Weights file (override default same name of .protobuf)')
    parser.add_argument('-in', dest='inputnode', type=str, help='Input node name')
    parser.add_argument('-on', dest='outputnode', type=str, help='Output node name')
    parser.add_argument('-s', dest='nshaves', type=int, default=1, help='Number of shaves (default 1)')
    parser.add_argument('-is', dest='inputsize', nargs=2, type=int, help='Input size for networks that don\'t provide an input shape, width and height expected')
    parser.add_argument('-dn', dest='device_no', metavar='', type=str, nargs='?', help="Experimental flag to run on a specified stick.")
    parser.add_argument('-ec', dest='explicit_concat', action='store_true', help='Force explicit concat')
    parser.add_argument('--accuracy_adjust', type=str, const="ALL:256", default="ALL:1", help='Scale the output by this amount', nargs='?')
    parser.add_argument('--ma2480', action="store_true", help="Dev flag")
    parser.add_argument('--scheduler', action="store", help="Dev flag")
    parser.add_argument('--new-parser', action="store_true", help="Dev flag")
    parser.add_argument('--model_dir',dest='model_dir', type=str, default='', help="Working directory in where to save outputs")
    args = parser.parse_args()
    return args


class Arguments:
    def __init__(self, network, inputnode, outputnode, inputsize, nshaves, weights, device_no, explicit_concat, ma2480, scheduler, model_dir, extargs):
        self.net_description = network
        filetype = network.split(".")[-1]
        self.parser = Parser.TensorFlow
        if filetype in ["prototxt"]:
            self.parser = Parser.Caffe
            if weights is None:
                weights = network[:-8] + 'caffemodel'
                if not os.path.isfile(weights):
                    weights = None
        self.conf_file = network[:-len(filetype)] + 'conf'
        if not os.path.isfile(self.conf_file):
            self.conf_file = None
        self.net_weights = weights
        self.input_node_name = inputnode
        self.output_node_name = outputnode
        self.input_size = inputsize
        self.number_of_shaves = nshaves
        self.image = 'Debug'
        self.raw_scale = 1
        self.mean = None
        self.channel_swap = None
        self.explicit_concat = explicit_concat
        self.acm = 0
        self.timer = True
        self.number_of_iterations = 1
        self.upper_temperature_limit = -1
        self.lower_temperature_limit = -1
        self.backoff_time_normal = -1
        self.backoff_time_high = -1
        self.backoff_time_critical = -1
        self.temperature_mode = 'Advanced'
        self.network_level_throttling = 1
        self.stress_full_run = 2
        self.stress_usblink_write = 1
        self.stress_usblink_read = 1
        self.debug_readX = 100
        self.mode = 'profile'
        self.outputs_name = model_dir+'output'
        self.save_input = None
        self.save_output = None
        self.device_no = device_no
        self.ma2480 = ma2480
        self.scheduler = scheduler
        self.new_parser = extargs.new_parser
        self.seed = -1
        self.accuracy_table = {}
        if extargs.accuracy_adjust != "":
            pairs = extargs.accuracy_adjust.split(',')
            for pair in pairs:
                layer, value = pair.split(':')
                self.accuracy_table[layer] = float(value)

def profile_net(network, inputnode = None, outputnode = None, nshaves = 1, inputsize = None, weights = None, device_no = None, explicit_concat = None, ma2480 = None, scheduler = True, model_dir = '', extargs = None):
    file_init()
    args = Arguments(network, inputnode, outputnode, inputsize, nshaves, weights, device_no, explicit_concat, ma2480, scheduler, model_dir, extargs)
    GLOBALS.USING_MA2480 = args.ma2480
    GLOBALS.OPT_SCHEDULER = args.scheduler is None
    ppInit(args.scheduler)
    myriad_config = load_myriad_config(args.number_of_shaves)

    if args.conf_file is not None:
        get_myriad_info(args, myriad_config)
    filetype = network.split(".")[-1]
    parser = None
    if filetype in ["prototxt"]:
        parser = Parser.Caffe
    elif filetype in ["pb", "protobuf", "meta"]:
        parser = Parser.TensorFlow
    else:
        throw_error(ErrorTable.ParserNotSupported)

    load_ret = load_network(args, parser, myriad_config,debug=True)
    net = load_ret['network']
    if args.new_parser:
        graph_file = load_ret['graph']
    else:
        graph_file = Blob([GLOBALS.BLOB_MAJOR_VERSION, GLOBALS.BLOB_MINOR_VERSION, GLOBALS.BLOB_PATCH_VERSION], net.name, '', myriad_config, net, model_dir+"graph")
        graph_file.generate_v2(args)
    timings, myriad_output = run_myriad(graph_file, args)
    net.gather_metrics(timings)
    print_summary_of_network(graph_file)
    if not args.new_parser:
        generate_graphviz(net, graph_file, filename=args.outputs_name)


if __name__ == "__main__":
    setup_warnings()

    print("\033[1mmvNCProfile v" + (u"{0:02d}".format(major_version, )) + "." +
          (u"{0:02d}".format(release_number, )) +
          ", Copyright @ Intel Corporation 2017\033[0m\n")
    args = parse_args()
    profile_net(args.network, args.inputnode, args.outputnode, args.nshaves, args.inputsize, args.weights, args.device_no, args.explicit_concat, args.ma2480, args.scheduler, args.model_dir, args)
