from logic.machine import Machine

m = Machine()
m.load_data("test.arff")
for i in range(10):
    m.learn()