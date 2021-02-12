import torch

eta = 0.0005
epochs = 10000

y_fn = lambda x: x**5 - x**4 - x
y_real = -1
# loss_fn = lambda y_real, y_predicted: y_predicted - y_real
loss_fn = lambda y_real, y_predicted: (y_predicted - y_real)**2/2
x_inits = [-2, -0.9, -0.5, 0, 0.5, 2]

results = []
for x_init in x_inits:
    x = torch.tensor(float(x_init), requires_grad=True)
    for i in range(epochs):
        y = y_fn(x)
        l = loss_fn(y_real, y)

        l.backward()

        if i <= 20 or i%100 == 0:
            print(f"i:{i:06d}/{epochs} x:{x:.6f} y:{y:.6f} l:{l:2.7f} grad:{x.grad:.6f}")

        x.data = x - eta*x.grad
        x.grad.zero_()

    print(f"We found x to be: {x}")
    results.append(f"x_init:{x_init:.4f} x:{x:.4f} y:{y:.4f} loss:{l:.8f}")

print("\nResults:")
print("\n".join(results))
print()


"""
(gdb) r ...
...
Results:
x_init:-2.0000 x:0.9278 y:-0.9813 loss:0.00017469
x_init:-0.9000 x:-1.0000 y:-1.0000 loss:0.00000000
x_init:-0.5000 x:0.9137 y:-0.9738 loss:0.00034214
x_init:0.0000 x:0.9211 y:-0.9779 loss:0.00024449
x_init:0.5000 x:0.9279 y:-0.9813 loss:0.00017428
x_init:2.0000 x:1.0478 y:-0.9902 loss:0.00004825


Thread 2 "python" received signal SIGSEGV, Segmentation fault.
[Switching to Thread 0x7fffe25ce640 (LWP 92182)]
0x00007ffff7dcc9aa in PyThreadState_Clear () from /usr/lib/libpython3.9.so.1.0
(gdb) where
#0  0x00007ffff7dcc9aa in PyThreadState_Clear () from /usr/lib/libpython3.9.so.1.0
#1  0x00007ffff5b7222c in pybind11::gil_scoped_acquire::dec_ref() ()
   from /home/user72/.local/lib/python3.9/site-packages/torch/lib/libtorch_python.so
#2  0x00007ffff5b72269 in pybind11::gil_scoped_acquire::~gil_scoped_acquire() ()
   from /home/user72/.local/lib/python3.9/site-packages/torch/lib/libtorch_python.so
#3  0x00007ffff5e89049 in torch::autograd::python::PythonEngine::thread_init(int, std::shared_ptr<torch::autograd::ReadyQueue> const&, bool)
    () from /home/user72/.local/lib/python3.9/site-packages/torch/lib/libtorch_python.so
#4  0x00007ffff54d36df in execute_native_thread_routine () from /home/user72/.local/lib/python3.9/site-packages/torch/lib/libtorch.so
#5  0x00007ffff79d63e9 in start_thread () from /usr/lib/libpthread.so.0
#6  0x00007ffff7aef293 in clone () from /usr/lib/libc.so.6
(gdb) exit
"""
