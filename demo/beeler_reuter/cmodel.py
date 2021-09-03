from ctypes import c_char_p
from ctypes import c_double
from ctypes import c_int
from typing import Dict
from typing import Optional

import numpy as np


class CModel:
    def __init__(self, lib, ode):
        self.lib = lib

        # Get number of states and parameters from the C library
        self.num_states = self.lib.state_count()
        self.num_parameters = self.lib.parameter_count()
        self.num_monitored = self.lib.monitor_count()
        self._init_lib()
        self.ode = ode

    def default_parameters(self):
        names = self.parameter_names()
        p = self.init_parameters()
        values = [p[self.parameter_index(name)] for name in names]
        return dict(zip(names, values))

    def parameter_names(self):
        return self.ode.parameter_symbols

    def state_names(self):
        return self.ode.state_symbols

    def monitor_names(self):
        return [
            expr.name for expr in self.ode.intermediates + self.ode.state_expressions
        ]

    def _init_lib(self):
        """
        Make sure that arrays passed to C is of the correct types.
        """

        float64_array = np.ctypeslib.ndpointer(
            dtype=c_double,
            ndim=1,
            flags="contiguous",
        )
        float64_array_2d = np.ctypeslib.ndpointer(
            dtype=c_double,
            ndim=2,
            flags="contiguous",
        )

        self.lib.init_state_values.restype = None  # void
        self.lib.init_state_values.argtypes = [float64_array]

        self.lib.init_parameters_values.restype = None  # void
        self.lib.init_parameters_values.argtypes = [float64_array]

        self.lib.state_index.restype = c_int
        self.lib.state_index.argtypes = [c_char_p]  # state_name

        self.lib.parameter_index.restype = c_int
        self.lib.parameter_index.argtypes = [c_char_p]  # state_name

        self.lib.monitored_index.restype = c_int
        self.lib.monitored_index.argtypes = [c_char_p]  # state_name

        self.lib.monitor.restype = None
        self.lib.monitor.argtypes = [float64_array_2d]  # states

        self.lib.monitored_values.restype = None
        self.lib.monitored_values.argtypes = [
            float64_array_2d,  # monitored
            float64_array_2d,  # states
            float64_array,  # parameters
            float64_array,  # u
            float64_array,  # m
            float64_array,  # t_values
            c_int,  # num_timesteps
        ]

        self.lib.monitored_values_single.restype = None
        self.lib.monitored_values_single.argtypes = [
            float64_array,  # monitored
            float64_array_2d,  # states
            float64_array,  # parameters
            float64_array,  # u
            float64_array,  # m
            float64_array,  # t_values
            c_int,  # num_timesteps
            c_int,  # index
        ]

        solve_functions = [
            self.lib.ode_solve_forward_euler,
            self.lib.ode_solve_rush_larsen,
        ]

        for func in solve_functions:
            func.restype = None  # void
            func.argtypes = [
                float64_array,  # u
                float64_array,  # parameters
                float64_array_2d,  # u_values
                float64_array,  # t_values
                c_int,  # num_timesteps
                c_double,  # dt
            ]

    def monitor(self, states, t, parameters=None):

        parameter_values = self.init_parameters()
        if parameters is not None:
            assert isinstance(parameters, dict)
            for pname, new_value in parameters.items():
                index = self.parameter_index(pname)
                old_value = parameter_values[index]
                if old_value != new_value:
                    parameter_values[index] = new_value

        u = np.zeros(self.num_states, dtype=np.float64)
        monitored_values = np.zeros((t.size, self.num_monitored), dtype=np.float64)
        m = np.zeros(self.num_monitored, dtype=np.float64)

        self.lib.monitored_values(
            monitored_values,
            states,
            parameter_values,
            u,
            m,
            t,
            t.size,
        )
        return monitored_values

    def monitor_single(self, name, states, t, parameters=None):
        index = self.monitor_index(name)
        parameter_values = self.init_parameters()
        if parameters is not None:
            assert isinstance(parameters, dict)
            for pname, new_value in parameters.items():
                pindex = self.parameter_index(pname)
                old_value = parameter_values[pindex]
                if old_value != new_value:
                    parameter_values[pindex] = new_value
        u = np.zeros(self.num_states, dtype=np.float64)
        monitored_values = np.zeros(t.size, dtype=np.float64)
        m = np.zeros(self.num_monitored, dtype=np.float64)

        self.lib.monitored_values_single(
            monitored_values,
            states,
            parameter_values,
            u,
            m,
            t,
            t.size,
            index,
        )
        return monitored_values

    def state_index(self, state_name: str) -> int:
        """Given a name of a state, return the index of it.

        Arguments
        ---------
        state_name : str
            Name of the state

        Returns
        -------
        int
            The index of the given state

        Note
        ----
        To list all possible states see `BaseModel.state_names`

        """
        assert isinstance(state_name, str)

        state_name_bytestring = state_name.encode()
        return self.lib.state_index(state_name_bytestring)

    def parameter_index(self, parameter_name: str) -> int:
        """Given a name of a parameter, return the index of it.

        Arguments
        ---------
        parameter_name : str
            Name of the parameter

        Returns
        -------
        int
            The index of the given parameter

        Note
        ----
        To list all possible tuning parameter see `BaseModel.parameter_names`

        """
        assert isinstance(parameter_name, str)

        parameter_name_bytestring = parameter_name.encode()
        return self.lib.parameter_index(parameter_name_bytestring)

    def monitor_index(self, monitor_name: str) -> int:
        """Given a name of a monitored expression, return the index of it.

        Arguments
        ---------
        monitor_name : str
            Name of the monitored expression

        Returns
        -------
        int
            The index of the given monitored expression

        Note
        ----
        To list all possible tuning parameter see `BaseModel.monitor_names`

        """
        assert isinstance(monitor_name, str)

        monitor_name_bytestring = monitor_name.encode()
        return self.lib.monitored_index(monitor_name_bytestring)

    def init_parameters(self):
        parameters = np.zeros(self.num_parameters, dtype=np.float64)
        self.lib.init_parameters_values(parameters)
        return parameters

    def default_inital_states(self):
        states = np.zeros(self.num_states, dtype=np.float64)
        self.lib.init_state_values(states)
        return states

    def solve(
        self,
        t_start,
        t_end,
        dt,
        num_steps=None,
        method="GRL1",
        u0: Optional[np.ndarray] = None,
        parameters: Optional[Dict[str, float]] = None,
    ):
        parameter_values = self.init_parameters()
        if parameters is not None:
            assert isinstance(parameters, dict)
            for name, new_value in parameters.items():
                index = self.parameter_index(name)
                old_value = parameter_values[index]
                if old_value != new_value:
                    parameter_values[index] = new_value
                    print(f"Update paramameter {name} from {old_value} to {new_value}")

        if type(dt) is not float:
            dt = float(dt)
        if num_steps is not None:
            assert type(num_steps) is int
            t_end = dt * num_steps
        else:
            num_steps = round((t_end - t_start) / dt)

        t_values = np.linspace(t_start, t_end, num_steps + 1)

        if u0 is not None:
            assert len(u0) == self.num_states
        else:
            u0 = np.zeros(self.num_states, dtype=np.float64)
            self.lib.init_state_values(u0)
        u_values = np.zeros((num_steps + 1, u0.shape[0]), dtype=np.float64)
        u_values[0, :] = u0[:]

        if method == "FE":
            self.lib.ode_solve_forward_euler(
                u0,
                parameter_values,
                u_values,
                t_values,
                num_steps,
                dt,
            )
        elif method == "GRL1":
            self.lib.ode_solve_rush_larsen(
                u0,
                parameter_values,
                u_values,
                t_values,
                num_steps,
                dt,
            )
        else:
            raise ValueError("Invalid method %s" % method)

        return t_values, u_values
