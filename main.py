"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 定义系统函数的分子和分母系数
numerator = [1, 5, -50]
denominator = [2, -2.98, 0.17, 2.3418, -1.5147]

# 计算零和极点
sys = signal.TransferFunction(numerator, denominator)
zeros = sys.zeros
poles = sys.poles

# 绘制零极点分布图
plt.figure()
plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='b', label='Zeros')
plt.scatter(np.real(poles), np.imag(poles), marker='x', color='r', label='Poles')

# 绘制单位圆
theta = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), 'k--', label='Unit Circle')

# 标出零点和极点坐标
for z in zeros:
    plt.annotate(f'({z.real:.2f}, {z.imag:.2f})', (z.real, z.imag), textcoords="offset points", xytext=(0, 10), ha='center')

for p in poles:
    plt.annotate(f'({p.real:.2f}, {p.imag:.2f})', (p.real, p.imag), textcoords="offset points", xytext=(0, 10), ha='center')

plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Zero-Pole Plot with Unit Circle')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.axis('equal')
plt.legend()
plt.show()
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 定义系统函数的分子和分母系数
numerator = [1, 5, -50]
denominator = [2, -2.98, 0.17, 2.3418, -1.5147]

# 定义输入信号（单位阶跃序列）
num_samples = 100
input_signal = np.ones(num_samples)

# 模拟系统的响应
system = signal.dlti(numerator, denominator, dt=1.0)
time, output_signal = signal.dlsim(system, input_signal)

# 绘制输入序列和输出序列图
plt.figure()
plt.stem(range(num_samples), input_signal, label='Input (Unit Step)')
plt.stem(range(num_samples), output_signal, markerfmt='ro', linefmt='r-', basefmt='r-', label='Output')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('System Response to Unit Step Input')
plt.legend()
plt.grid()
plt.show()








