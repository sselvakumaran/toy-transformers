# NEW VERSION

import matplotlib.pyplot as plt
import seaborn as sns
import bisect
from itertools import zip_longest
from typing import Optional, List, Literal, Tuple

# store 2d points easier
# assumes inputted in non-decreasing order
class _PointHandler:
	def __init__(self):
		self.xs = []
		self.ys = []
	
	def __bool__(self):
		return len(self.xs) > 0 and len(self.ys) > 0

	def add_point(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
		if y is None:
			return # convenient to put in optional arguments without case
		x = x if x is not None else self.get_last_point() + 1
		self.xs.append(x)
		self.ys.append(y)
	
	def add_points(self, xs: List[int], ys: List[int]) -> None:
		if len(xs) != len(ys):
			raise ValueError("lists are not of equal length")
		self.xs.extend(xs)
		self.ys.extend(ys)
	
	def get_last_point(self) -> Tuple[int, int]:
		if not self:
			return None
		return (self.xs[-1], self.ys[-1])
	
	def get_last_points(self, 
		n: int = 50, 
		by: Literal['x', 'index'] = 'index',
		include_previous: bool = True # include one more by index
	):
		if not self:
			return ([], [])
		
		if by == 'index':
			start_idx = n + (1 if include_previous else 0)
			return(self.xs[-start_idx:], self.ys[-start_idx:])
		
		# by == 'x'
		start_val = self.get_last_point()[0] - (n - 1)
		start_idx = bisect.bisect_left(self.xs, start_val)

		return (self.xs[start_idx:], self.ys[start_idx:])


# TODO: fix
class LossTracker:
	# initialize settings
	def __init__(self,
			palette: str = "colorblind"
		):
		sns.set_style("ticks")
		sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2})
		sns.set_palette(palette)
		self.fig = plt.figure()

		self.x = []
		self.train = []
		self.val = []
		self.last_train_x = 0
		self.last_val_x = 0
	
	# Add singular value to data tracker, assumes called in non-decreasing order
	def add_points(self, 
		train: float, 
		val: Optional[float] = None, 
		x: Optional[int] = None
	):
		current_x = x or (self.last_train_x + 1)

		self.x.append(current_x)
		self.train.append(train)
		self.last_train_x = max(self.last_train_x, current_x)

		self.val.append(val)
		if val is not None:
			self.last_val_x = max(self.last_val_x, current_x)

	# Add batch of values to data tracker 
	def add_points_batch(self, 
		trains: List[float], 
		vals: List[Optional[float]] = None,
	):
		for train, val in zip_longest(trains, vals):
			self.add_points(train, val)

	def render_fast_figure(self) -> plt.Figure:
		self.fig.clear()

		ax1 = self.fig.add_subplot(1, 2, 1)
		sns.lineplot(x=self.x, y=self.train, ax=ax1, alpha=0.6)
		sns.lineplot(x=self.x, y=self.val,   ax=ax1)

		if self.train:
			ax1.scatter(self.last_train_x, self.train[-1], s=40, zorder=3)
		if self.val:
			ax1.scatter(self.last_val_x, self.val[self.last_val_x - 1], s=40, zorder=3)
		
		ax2 = self.fig.add_subplot(1, 2, 2)
		ax2_x = self.x[-50:]
		sns.lineplot(x=ax2_x, y=self.train[-50:], ax=ax2, alpha=0.6)
		sns.lineplot(x=ax2_x, y=self.val[-50:],   ax=ax2)

		sns.despine(fig=self.fig)

		return self.fig

	# Render and return figure to print
	def render_full_figure(self) -> plt.Figure:
		pass
		# if self.y_train:
		# 	if self.train_line:
		# 		self.train_line.set_data(self.x, self.y_train)
		# 	else:
		# 		sns.lineplot(x=self.x, y=self.y_train, ax=self.ax1)
		# 		self.train_line = self.ax1.lines[0]

		# if any(y is not None for y in self.y_val):
		# 	if self.val_line:
		# 		self.val_line.set_data(self.x, self.y_val)
		# 	else:
		# 		sns.lineplot(x=self.x, y=self.y_val, ax=self.ax1)
		# 		self.val_line = self.ax1.lines[1]
		
		# self.ax1.relim()
		# self.ax1.autoscale_view()

		# self.fig.canvas.draw_idle()
		# return self.fig