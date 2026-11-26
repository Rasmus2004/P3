# -----------------------------------------------------------------------------
# Copyright (c) 2026, Lucid Vision Labs, Inc.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# -----------------------------------------------------------------------------


import time
from arena_api.system import system
import textwrap

"""
Explore: Node Types
    This example explores the properties of the various node types
    including boolean, string, enumeration, integer, and float nodes. The user
    inputs the node name that they wish to access (leaving out spacing between
    words) in order to retrieve the node properties, or inputs 'x' to exit.
    See Explore_Nodes for a complete list of nodes and their respective types.
"""

"""
 =-=-=-=-=-=-=-=-=-
 =-=- EXAMPLE -=-=-
 =-=-=-=-=-=-=-=-=-
"""
TAB1 = "  "
TAB2 = "    "
TAB6 = "            "

def create_devices_with_tries():
    """
    Waits for the user to connect a device before raising an exception
    if it fails
    """
    tries = 0
    tries_max = 6
    sleep_time_secs = 10
    devices = None
    while tries < tries_max:  # Wait for device for 60 seconds
        devices = system.create_device()
        if not devices:
            print(
                f'{TAB1}Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
                f'secs for a device to be connected!')
            for sec_count in range(sleep_time_secs):
                time.sleep(1)
                print(f'{TAB1}{sec_count + 1 } seconds passed ',
                      '.' * sec_count, end='\r')
            tries += 1
        else:
            print(f'{TAB1}Created {len(devices)} device(s)')
            return devices
    else:
        raise Exception(f'{TAB1}No device found! Please connect a device and run '
                        f'the example again.')


def explore_boolean(node):
    """
    explores nodes of boolean type
    (1) retrieves value
    """
    print(f"{TAB2}Value: {node.value}")


def explore_string(node):
    """
    explores nodes of string type
    (1) retrieves value
    (2) retrieves maximum value length
    """
    print(f"{TAB2}Value: {node.value}")
    print(f"{TAB2}Maximum Length: {node.max}")


def explore_integer(node):
    """
    explores nodes of type integer
    (1) retrieves value
    (2) retrieves minimum and maximum
    (3) retrieves increment and increment mode
    (4) retrieves representation
    (5) retrieves unit
    """
    print(f"{TAB2}Value: {node.value}")
    print(f"{TAB2}Minimum: {node.min}, Maximum: {node.max}")
    print(f"{TAB2}Increment Mode: {node.inc} ({str(node.inc_mode)})")
    print(f"{TAB2}Representation: {str(node.representation)}")
    print(f"{TAB2}Unit: {node.unit}")


def explore_float(node):
    """
    explores nodes of type integer
    (1) retrieves value
    (2) retrieves minimum and maximum
    (3) retrieves increment and increment mode
    (4) retrieves representation
    (5) retrieves unit
    (6) retrieves display notation
    (7) retrieves display precision
    """
    print(f"{TAB2}Value: {node.value}")
    print(f"{TAB2}Minimum: {node.min}, Maximum: {node.max}")

    if node.inc is not None:
        print(f"{TAB2}Increment Mode: {node.inc} ({str(node.inc_mode)})")

    print(f"{TAB2}Representation: {str(node.representation)}")
    print(f"{TAB2}Unit: {node.unit}")
    print(f"{TAB2}Display Notation: {str(node.display_notation)}")
    print(f"{TAB2}Display Precision: {node.display_precision}")


def explore_enumeration(node):
    """
    explores nodes of string type
    (1) retrieves value
    (2) retrieves entries
    """


    print(f"{TAB2}Current Entry: {node.value}")
    print(f"{TAB2}Integer Value: {node.enumentry_nodes[node.value].int_value}")
    print(f"{TAB2}Entries: {str(node.enumentry_names[0])}")
    i = 1
    while i < len(node.enumentry_names):
        print(f"{TAB6} {str(node.enumentry_names[i])}")
        i += 1



def explore_nodes(nodemap):
    """
    controls node exploration
    """
    node_name = input(f"{TAB1}Input node name to explore from Device Nodemap ('x' to exit): ")

    # stay in loop until exit
    while True:
        # exit manually on 'x'
        if node_name.__eq__('x'):
            print(f"{TAB1}Succesfully exited")
            break

        # get node
        try:
            node = nodemap.get_node(str(node_name))

        # explore by type
            if node and node.is_readable:
                if node.interface_type.value == 3:
                    explore_boolean(node)
                elif node.interface_type.value == 6:
                    explore_string(node)
                elif node.interface_type.value == 9:
                    explore_enumeration(node)
                elif node.interface_type.value == 2:
                    explore_integer(node)
                elif node.interface_type.value == 5:
                    explore_float(node)
                else:
                    print(f"{TAB2}Type not found")

        except ValueError as e:
            error_message = str(e)
            error_message = error_message.replace("\n", f"\n{TAB2}")
            print(f'{TAB2}{error_message}')

        except:
            print(f"{TAB2}Node not found")

        # reset input
        node_name = ""
        node_name = input(f"{TAB1}Input node name to explore from Device Nodemap ('x' to exit): ")


"""
 =-=-=-=-=-=-=-=-=-
 =- PREPARATION -=-
 =- & CLEAN UP =-=-
 =-=-=-=-=-=-=-=-=-
"""


def example_entry_point():
    # prepare example
    devices = create_devices_with_tries()
    device = system.select_device(devices)

    nodemap = device.nodemap

    # run example
    explore_nodes(nodemap)

    # clean up example
    print(f"{TAB1}Destroy Device")
    system.destroy_device(device)


if __name__ == "__main__":
    print(f"Example Started\n")
    example_entry_point()
    print(f"\nExample Completed")
