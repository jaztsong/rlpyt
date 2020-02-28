from visdom import Visdom
import numpy as np

class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name='main'):
        self.viz = Visdom()
        assert(self.viz.check_connection())
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y, update='append'):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array(x), Y=np.array(y), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='X-axis',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array(x), Y=np.array(y), env=self.env,
                          win=self.plots[var_name], name=split_name, update=update)
