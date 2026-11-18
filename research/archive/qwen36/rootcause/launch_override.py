"""Process-local launch-parameter override around an unchanged official Triton kernel."""
class LaunchOverride:
    def __init__(self, kernel, num_warps):
        if num_warps not in (1, 2, 4): raise ValueError('Unsupported diagnostic warp count')
        self.kernel=kernel;self.num_warps=num_warps
    def __getitem__(self, grid):
        launch=self.kernel[grid]
        def run(*args,**kwargs):
            kwargs=dict(kwargs);kwargs['num_warps']=self.num_warps
            return launch(*args,**kwargs)
        return run
