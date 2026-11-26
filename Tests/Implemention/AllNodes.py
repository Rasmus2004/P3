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

'''
Explore: Nodes
        This example explores traversing the nodes as a tree and fundamental node
        information including display name, access mode visibility, interface type,
        and value.
'''

'''
=-=-=-=-=-=-=-=-=-
=-=- SETTINGS =-=-
=-=-=-=-=-=-=-=-=-
'''
TAB1 = "  "
TAB2 = "    "

# Choose node properties to explore
explore_access = True
explore_visibility = True
explore_type = True
explore_value = True

'''
=-=-=-=-=-=-=-=-=-
=-=- EXAMPLE -=-=-
=-=-=-=-=-=-=-=-=-
'''


def create_devices_with_tries():
        '''
        Waits for the user to connect a device before raising
                an exception if it fails
        '''
        tries = 0
        tries_max = 6
        sleep_time_secs = 10
        devices = None
        while tries < tries_max:
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


def explore_node(node, nodemap, depth):
        '''
        explores node
        (1) retrieves display name
        (2) retrieves node name
        (3) retrieves accessibility
        (4) retrieves visibility
        (5) retrieves interface type
        (6) retrieves value
        '''
        # Get display name
        display_name = node.display_name

        # Get node name
        node_name = node.name

        # Retrieve accessibility
        access_mode = node.access_mode
        access_mode_str = str(access_mode)

        # Retrieve visibility
        visibility = node.visibility
        visibility_str = str(visibility)

        # Retrieve interface type
        interface_type = node.interface_type
        interface_type_str = str(interface_type)

        # Retrieve value if node is not a category or register node and is readable
        value = "-"
        if node.is_readable and node.interface_type.value != 7 and node.interface_type.value != 8:
                value = str(node.value)

        # Check and print the desired information
        access_mode_explored = None
        visibility_explored = None
        interface_type_explored = None

        if explore_access:
                access_mode_explored = access_mode_str

        if explore_visibility:
                visibility_explored = visibility_str

        if explore_type:
                interface_type_explored = interface_type_str

        if explore_value:
                value_explored = value if len(value) < 30 else "..."

        display_and_node_name = display_name + " (" + node_name + ")"

        print(TAB1 * depth, '{:<{name_width}}{:<20}{:<25}{:<30}{:}'.format(display_and_node_name,
                access_mode_explored, visibility_explored, interface_type_explored,
                 value_explored, name_width = 90 - depth*2))

        # Explore category node children
        if node.interface_type.value == 8 and node.is_readable:
                children = node.features
                for val in children:
                        explore_node(nodemap.get_node(val), nodemap, depth+1)


def example_entry_point(device):
        # Initialize the Root nodes from all nodemaps, call the explore nodes function
        nodemaps = [device.nodemap, device.tl_device_nodemap, device.tl_stream_nodemap,
                                device.tl_interface_nodemap, system.tl_system_nodemap]
        nodemap_names = ["Device Nodemap", "TL Device Nodemap", "TL Stream Nodemap",
                                         "TL Interface Nodemap", "TL System Nodemap" ]

        for i, nodemap in enumerate(nodemaps):
                print(f"\n{TAB1}{nodemap_names[i]}")
                explore_node(nodemap.get_node("Root"), nodemap, 1)

        # Cleanup
        system.destroy_device()


if __name__ == "__main__":
        print("Example Started\n")
        devices = create_devices_with_tries()
        device = system.select_device(devices)

        example_entry_point(device)
        print("\nExample Finished")
