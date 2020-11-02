import os
from collections import defaultdict

import numpy as np

from . import make_logger

logger = make_logger(__name__, 10)


def strain_dict_to_list(strain_data=None):
    """
    Strain data are organized in a dictionary where
    the first key is the region, and each region contains
    list or strain data for each time and at each time
    there is a triplet (circumferential, radial, longitudinal)
    strain_data[region][timepoint]

    This functon converts this dictionary to a list
    so that if s = strain_data[timepoint] then s is a
    dictionary with keys s[region]
    """

    if strain_data is None:
        return strain_data

    # First find the number of timepoints (assuming this is consitent)
    num_points = len(strain_data.values()[0])

    new_strain_data = []

    for time_index in range(num_points):

        this_strain_data = defaultdict(float)

        for region, time_data in strain_data.items():

            this_strain_data[region] = time_data[time_index]

        new_strain_data.append(this_strain_data)

    return tuple(new_strain_data)


class Observation(object):
    def __init__(self, real_observation, model_observation):
        self.real_observation = real_observation
        self.model_observation = model_observation


class Observations(object):
    """"""

    _target_names = ("strain", "volume")
    _data_keys = ("time", "volume", "strain", "RVV")

    def __init__(
        self,
        bcs=None,
        time=None,
        volume=None,
        strain=None,
        RVV=None,
        RVP=None,
        # start_passive=None,
        # start_active=None,
        echo_data=None,
        **kwargs
    ):

        self.bcs = bcs
        self.time = time
        self.volume = volume
        self.RVV = RVV
        self._init_strain(strain)
        self.additional_data = kwargs
        self.echo_data = echo_data

        self._check_data()
        self._check_targets()

        if self.time is None:
            self.time = np.arange(self.num_points)
        else:
            # Make first time point zero
            self.time = tuple(np.subtract(self.time, self.time[0]).tolist())

        # self.start_passive = start_passive \
        #     if start_passive is not None else 0
        # self.start_active = start_active \
        #     if start_active is not None else self.start_passive + 1

        # self.passive_duration = self.start_active - self.start_passive
        # self.num_contract_points = self.num_points - self.passive_duration

        self.current_point = 0
        self._tupleize()

    def _tupleize(self):
        """
        Make data immutable by converting them to tuples
        """
        for k in self._data_keys:
            v = getattr(self, k)
            if v is None:
                continue
            if isinstance(v, np.ndarray):
                v = v.tolist()
            setattr(self, k, tuple(v))

    def __repr__(self):
        args = []
        for attr in ("time", "pressure", "volume", "strain", "echo_data"):
            if getattr(self, attr) is not None:
                args.append(attr)

        return ("{self.__class__.__name__}" "({args})").format(
            self=self, args=", ".join(args)
        )

    def _init_strain(self, strain):

        self.strain_dict = strain
        self.strain = strain_dict_to_list(strain)

    def _check_data(self):

        data = self.data_dict
        assert len(data.values()) > 0, "Please provide some data"
        pts = [len(q) for q in data.values()]
        num_points = pts[0]
        msg = ("Size of data is not consistent. \n" "{}").format(
            "\n".join(["{}: {}".format(k, len(v)) for k, v in data.items()])
        )
        assert len(set(pts)) == 1, msg
        self.num_points = num_points
        return data

    def __getitem__(self, attr):
        # attr, it = args

        # return dolfin_adjoint.Constant(getattr(self, attr)[it])
        return getattr(self, attr)

    @property
    def data_dict(self):
        """
        Collect all data into a dictionary
        """
        data = {}

        def append_data(d, key):

            if isinstance(d, (list, tuple)):
                data[key] = tuple(d)

            elif isinstance(d, dict):
                for k, v in d.items():
                    append_data(v, "{}_{}".format(key, k))

            elif isinstance(d, np.ndarray):
                data[key] = tuple(d.tolist())

            else:
                msg = "Unknown data type {}".format(type(d))
                raise ValueError(msg)

        for attr in self._data_keys:

            d = getattr(self, attr)

            if d is None:
                continue

            append_data(d, attr)
        return data

    def _check_targets(self):

        self.targets = []
        for attr in self._target_names:

            if hasattr(self, attr):
                self.targets.append(attr)

    @classmethod
    def from_file(cls, data_path=None, echo_path=None):

        if data_path is not None:
            name, ext = os.path.splitext(data_path)
            extensions = ".yml"
            # extensions = ('.npy', '.yml', '.json',
            # '.xls', '.xlsx', '.csv')
            msg = (
                "Filename has to have one of the following " "extensions {}, got {}"
            ).format(extensions, ext)
            assert ext in extensions, msg

            data_kwargs = Observations.load_yaml_file(data_path)
        else:
            data_kwargs = {}

        if echo_path is not None:
            echo_data = Observations.load_echo_data(echo_path)

        else:
            echo_data = None

        return cls(echo_data=echo_data, **data_kwargs)

    def to_yaml(self, fname):

        data = self.data_dict
        # data['start_passive'] = self.start_passive
        # data['start_active'] = self.start_active

        if "strain" in data:
            data["strain"] = self.strain_dict

        try:
            import yaml

            with open(fname, "w") as f:
                yaml.dump(data, f)
        except ImportError as ex:
            logger.error("Please install yaml. Pip install pyyaml")
            raise ex
        except Exception as ex:
            logger.error("Cannot save data to {}".format(fname))
            logger.error(("Data contains the following keys " "{}").format(data.keys()))

            raise ex

    @staticmethod
    def load_echo_data(fname):
        if fname == "":
            return {}

        logger.debug("Load echo data from file {}".format(fname))
        pass

    @staticmethod
    def load_yaml_file(fname):

        if fname == "":
            return {}

        try:
            import yaml

            with open(fname, "r") as f:
                data = yaml.load(f)
            logger.debug("Load data from file {}".format(fname))

        except ImportError as ex:
            logger.error(("You need to install yaml. " "pip install pyyaml"))
            raise ex

        except IOError as ex:
            logger.error("File {} does not exist".format(fname))
            raise ex

        else:

            if "start_passive" not in data:
                data["start_passive"] = data.pop("passive_filling_begins", 0)

            if "start_active" not in data:
                data["start_active"] = data["start_passive"] + data.pop(
                    "passive_filling_duration", 1
                )

            return data

    # @property
    # def active_times(self):
    #     return self.time[self.passive_duration:]

    def __iter__(self):
        return self

    def next(self):

        if self.current_point == self.num_points:
            raise StopIteration

        else:
            p = self.current_point
            current_time = self.time[p]
            # current_pressure = self.pressure[p]

            # current_volume = None if self.volume \
            #     is None else self.volume[p]

            # current_strain = None if self.strain \
            #     is None else self.strain[p]

            # self.current_point += 1
            # return (current_time,
            #         current_pressure,
            #         current_volume,
            #         current_strain)
            return current_time

    def interpolate_data(self, start, n=1):
        """Interpolate data for the pressure,
        volume and strain between start and start +1,
        and return n new points between to
        successive ones
        """

        if n == 0:
            return

        # Possible meausurements
        attrs = ["time", "volume", "pressure", "strain", "RVV", "RVP"]

        # The original data at the integers
        xp = np.arange(self.num_points)
        # Add x-values for where to interpolate
        x = sorted(xp + np.linspace(start, start + 1, n + 2)[1:-1].tolist())

        for att in attrs:

            if not hasattr(self, att):
                continue

            arr = getattr(self, att)

            if arr is None:
                continue

            if att == "strain":

                strain = {}
                for r, s in self.strain_dict.items():
                    f0 = np.interp(x, xp, np.transpose(s)[0]).tolist()
                    f1 = np.interp(x, xp, np.transpose(s)[1]).tolist()
                    f2 = np.interp(x, xp, np.transpose(s)[2]).tolist()
                    strain[r] = zip(f0, f1, f2)
                self._init_strain(strain)

            else:
                arr_int = np.interp(x, xp, arr).tolist()
                setattr(self, att, arr_int)

        if start < self.passive_duration:
            self.passive_filling_duration += n
            self.num_points += n
        else:
            self.num_contract_points += n
            self.num_points += n

        self.number_of_interpolations += n
