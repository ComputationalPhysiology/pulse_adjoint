import dolfin_adjoint

def make_functional():

    # Get the functional value of each term in the functional
    func_lst = []
    for key,val in self.target_params.iteritems():
        if val:
            func_lst.append(self.opt_weights[key]*self.optimization_targets[key].get_functional())
            
        # Collect the terms in the functional
        functional = list_sum(func_lst)
        # Add the regularization term
        functional += self.regularization.get_functional()

        return functional

def update_targets(self, it, u, m, annotate=False):

    for key,val in self.target_params.iteritems():
        
        if val:
                       
            self.optimization_targets[key].next_target(it, annotate=annotate)
            self.optimization_targets[key].assign_simulated(u)
            self.optimization_targets[key].assign_functional()
            self.optimization_targets[key].save()

                        
        self.regularization.assign(m, annotate = annotate)
        self.regularization.save()

def reset_targets(assimilator):
    # Set the functional value for each target to zero
    for key, val in assimilator.target_params.items():
        if val:
            assimilator.optimization_targets[key].reset()

    assimilator.regularization.reset()


def next_target(assimilator):
    for key, val in assimilator.target_params.items():
        if val:
            assimilator.optimization_targets[key].\
                next_target(0, annotate=annotate)
            

def increment_time_step(assimilator):

    if assimilator.phase == "active":
        # There is only on step, so we are done
        functionals_time.append(functional*dt[START_TIME])
        adj_inc_timestep(1, True)
                    
    else:
        # Check if we are done with the passive phase
        functionals_time.append(functional*dt[it])
        dolfin_adjoint.\
            adj_inc_timestep(it, it == assimilator.current_steps.number)
        
        functional_values.append(dolfin.assemble(functional))

        
def solve_forward_problem1(assimilator, annotate=False):

    assimilator.reset_target()

    # Start the clock
    dolfin_adjoint.adj_start_timestep(0.0)

    states = []
    functional_values = []
    functionals_time = []

    if assimilator.phase == "passive":
        assimilator.next_target()
        # And we save it for later reference
        assimilator.problem.solve()
        states.append(assimilator.problem
                      .state.copy(deepcopy=True))

    # Print the functional
    # logger.info(self._print_functional())
    # Print the head of table
    # logger.info(self._print_head())

    functional = assimilator.functional

    if assimilator.phase == "passive":

        # Add the initial state to the recording
        functionals_time.append(functional*dolfin_adjoint.dt[0.0])

    for it, (step, state) in enumerate(assimilator.current_steps):

        states.append(state.copy(deepcopy=True))

        # Some check to see if we should annotate
        if not assimilator.annotation_step(step):
            continue

        assimilator.update_targets(annotate=annotate)

        # Print the values
        # logger.info(self._print_line(it))

        assimilator.increment_time_step()

        forward_result = make_forward_result(functional_values,
                                             functionals_time)

        # self._print_finished_report(forward_result)
        return forward_result

