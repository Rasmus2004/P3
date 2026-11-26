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
Explore: Node Maps
        This example explores the 5 available node maps of Arena, 4 retrieved from
        any devices and 1 from the system. It demonstrates traversing nodes retrieved
        as a complete vector.
'''
TAB1 = "  "
TAB2 = "    "
'''
=-=-=-=-=-=-=-=-=-
=-=- EXAMPLE =-=-
=-=-=-=-=-=-=-=-=-
'''


def create_devices_with_tries():
        '''
        This function waits for the user to connect a device before raising
                an exception
        '''

        tries = 0
        tries_max = 6
        sleep_time_secs = 10
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


def explore_nodemap(nodemap, nodemap_name):
        '''
        This function gets and prints all the features of the
                nodemap into the terminal and a file
        '''
        feature_nodes_names = nodemap.feature_names
        print(f'{TAB2}Number of nodes: {len(feature_nodes_names)}')

        list = []

        for node_name in feature_nodes_names:
                if nodemap[node_name].interface_type.name == 'CATEGORY' and nodemap[node_name].is_readable:
                        list.append(nodemap[node_name].name)

        with open(f'arena_api_node_exploration_{nodemap_name}.txt', 'w') as f:
                for node_name in feature_nodes_names:
                        # print to file
                        print(nodemap[node_name], file=f)

        print(f'{TAB2}Category nodes: ' + str(list))


def example_entry_point():
        '''
        explores: node maps
        (1) retrieves node map from device
        (2) retrieves node maps from corresponding transport layer modules
        (3) explores the device node map
        (4) explores the transport layer device node map
        (5) explores the transport layer stream node map
        (6) explores the transport layer interface node map
        (7) explores the transport layer system node map
        '''
        # Create a device
        devices = create_devices_with_tries()
        device = system.select_device(devices)

        explore_device = True
        explore_tl_device = True
        explore_tl_stream = True
        explore_tl_interface = True
        explore_tl_system = True

        # Explore nodemaps --------------------------------------------------------
        if explore_device:
                print(f'{TAB1}Exploring Device nodemap')
                explore_nodemap(device.nodemap, 'device_nodemap')

        if explore_tl_device:
                print(f'{TAB1}Exploring Transport Layer Device nodemap')
                explore_nodemap(device.tl_device_nodemap, 'TL_device_nodemap')

        if explore_tl_stream:
                print(f'{TAB1}Exploring Transport Layer Stream nodemap')
                explore_nodemap(device.tl_stream_nodemap, 'TL_stream_nodemap')

        if explore_tl_interface:
                print(f'{TAB1}Exploring Transport Layer Interface nodemap')
                explore_nodemap(device.tl_interface_nodemap, 'TL_interface_nodemap')

        if explore_tl_system:
                print(f'{TAB1}Exploring Transport Layer System nodemap')
                explore_nodemap(system.tl_system_nodemap, 'TL_system_nodemap')

        # Clean up ---------------------------------------------------------------

        # Destroy device.
        system.destroy_device()


if __name__ == '__main__':
        print('Example started\n')
        example_entry_point()
        print('\nExample finished successfully')
