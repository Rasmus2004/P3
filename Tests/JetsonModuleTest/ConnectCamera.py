from  arena_api.system import system

device = system.create_device()[0]
print(f"Conncted to device: {device}")

system.destroy_device()
