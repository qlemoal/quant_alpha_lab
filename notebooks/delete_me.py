import setup
from src.utils.my_plotting import new_fig, finish_fig
import matplotlib.pyplot as plt
import sys

print(sys.path)

f, ax = new_fig()
plt.plot([1, 2, 3], [4, 6, 2])
finish_fig(ax, title='Hello world', xl='Hello', yl='World')




