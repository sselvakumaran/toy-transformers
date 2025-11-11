import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, clear_output

class LossPlotter:
	def __init__(self,
		title="loss",
		style="darkgrid"
	):
		sns.set_style(style)
		self.train_losses = []
		self.val_losses = []
		self.iterations = []

		self.fig, self.ax = plt.subplots(figsize=(8, 5))
		self.title = title
		plt.ion()
	
	def update(self,
		train_loss: float, 
		val_loss: float = None,
		iteration: int = None
	):
		iteration = iteration or len(self.train_losses) + 1
		self.train_losses.append(train_loss)
		self.iterations.append(iteration)
		if val_loss is not None:
			self.val_losses.append((iteration, val_loss))
		
		self._draw_plot()

	def _draw_plot(self):
		clear_output(wait=True)
		fig, ax = plt.subplots(figsize=(8,5))

		sns.lineplot(
			x = self.iterations,
			y = self.train_losses,
			label = "train",
			ax = ax
		)

		if self.val_losses:
			val_iters, vals = zip(*self.val_losses)
			sns.lineplot(
				x = val_iters, 
				y = vals,
				label = "val",
				ax = ax
			)
		
		ax.set_title(self.title)
		ax.set_xlabel("iteration")
		ax.set_ylabel("loss")

		latest_train_x = self.iterations[-1]
		latest_train_y = self.train_losses[-1]
		ax.text(
			latest_train_x + 0.5,
			latest_train_y,
			f"{latest_train_y:.4f}",
			color="tab:blue",
			va="center"
		)

		ax.legend()
		#fig.canvas.draw()
		#self.fig.canvas.flush_events()

		display(fig)
		plt.close(fig)

	def show(self):
		plt.ioff()
		plt.show()