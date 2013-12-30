import pyopencl as cl
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

print " "
print "Getting platform info... "
print "Platform info: "
for found_platform in cl.get_platforms():
    print found_platform
    my_platform = found_platform

print " "
print "Getting device info... "
print "Device info: "
for found_device in my_platform.get_devices():
    print found_device
print " "

class CL:
    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

    def loadProgram(self, filename):
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        self.program = cl.Program(self.ctx, fstr).build()

    def memAlloc(self):
        mf = cl.mem_flags

        self.a = np.array(np.linspace(-1, 1, 10000), dtype=np.float32)

        self.a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.a) # parameter
        self.dest1_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.a.nbytes) # z
        self.dest2_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.a.nbytes) # x
        self.dest3_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.a.nbytes) # y

    def execute(self):
        self.program.kernels(self.queue, self.a.shape, None, self.a_buf, self.dest1_buf, self.dest2_buf, self.dest3_buf)
        b = np.empty_like(self.a)
        c = np.empty_like(self.a)
        d = np.empty_like(self.a)
        cl.enqueue_read_buffer(self.queue, self.dest1_buf, b).wait()
        cl.enqueue_read_buffer(self.queue, self.dest2_buf, c).wait()
        cl.enqueue_read_buffer(self.queue, self.dest3_buf, d).wait()

        mpl.rcParams['legend.fontsize'] = 10
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(c, d, b, label='parametric curve')
        ax.legend()
        plt.show()
        plt.savefig('test.pdf', format='pdf')

if __name__ == "__main__":
    example = CL()
    example.loadProgram("kernels.cl")
    example.memAlloc()
    example.execute()


