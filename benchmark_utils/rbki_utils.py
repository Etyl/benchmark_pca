from benchopt.stopping_criterion import StoppingCriterion

class RBKICriterion(StoppingCriterion):
    # very hacky, as I can't rely neither on callback approach neither on global max_runs 
    def get_runner_instance(self, max_runs=1, timeout=None, output=None, solver=None):
        criterion = super().get_runner_instance(max_runs, timeout, output, solver)
        if not hasattr(solver, "max_runs"):
            raise TypeError("Solver should have max_runs attribute")
        criterion.max_runs = solver.max_runs
        return criterion 
    
    def get_next_stop_val(self, stop_val):
        return stop_val + 1
    
    def init_stop_val(self):
         return 1

    def check_convergence(self, objective_list):
        i = len(objective_list) - 1
        return i == self.max_runs, i / self.max_runs