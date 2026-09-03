from configparser import ConfigParser
from collections import OrderedDict
from itertools import chain, permutations
from functools import cache

from yaml import load as yaml_load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import numpy as np

from spanet.dataset.types import *
from spanet.network.utilities.group_theory import (
    power_set,
    complete_symbolic_symmetry_group,
    complete_symmetry_group,
    expand_permutations,
    create_group_string
)


def cached_property(func):
    return property(cache(func))


def with_default(value, default):
    return default if value is None else value


def key_with_default(database, key, default):
    if key not in database:
        return default

    value = database[key]
    return default if value is None else value


class EventInfo:
    def __init__(
        self,

        # Information about observable inputs for this event.
        input_types: InputDict[str, InputType],
        input_features: InputDict[str, Tuple[FeatureInfo, ...]],

        # Information about the target structure for this event.
        event_particles: Particles,
        product_particles: EventDict[str, Particles],

        # Information about auxiliary values attached to this event.
        regressions: FeynmanDict[str, List[RegressionInfo]],
        classifications: FeynmanDict[str, List[ClassificationInfo]]
    ):

        self.input_types = input_types
        self.input_names = list(input_types.keys())
        self.input_features = input_features

        self.event_particles = event_particles
        self.event_mapping = self.construct_mapping(self.event_particles)
        self.event_symmetries = Symmetries(
            len(self.event_particles),
            self.apply_mapping(self.event_particles.permutations, self.event_mapping)
        )

        self.product_particles = product_particles
        self.product_mappings: ODict[str, ODict[str, int]] = OrderedDict()
        self.product_symmetries: ODict[str, Symmetries] = OrderedDict()

        for event_particle, product_particles in self.product_particles.items():
            product_mapping = self.construct_mapping(product_particles)

            self.product_mappings[event_particle] = product_mapping
            self.product_symmetries[event_particle] = Symmetries(
                len(product_particles),
                self.apply_mapping(product_particles.permutations, product_mapping)
            )

        self.regressions = regressions
        self.classifications = classifications

        self.validate_product_sources()

    def validate_product_sources(self):
        """ Make sure that the input assignments of the products are compatible with the symmetries.

        Two products which may be exchanged by a symmetry have to be indistinguishable, so they must
        also come from the same input collection. The same holds for two event particles which may be
        exchanged by an event-level symmetry.
        """
        for event_particle, product_particles in self.product_particles.items():
            sources = product_particles.sources
            permutations = self.product_symmetries[event_particle].permutations

            for permutation in permutations:
                for cycle in permutation:
                    cycle_sources = {sources[index] for index in cycle}
                    if len(cycle_sources) > 1:
                        cycle_names = ", ".join(product_particles[index] for index in cycle)
                        raise ValueError(
                            f"Products ({cycle_names}) of {event_particle} are related by a symmetry "
                            f"but are assigned to different inputs. "
                            f"Symmetric products must share the same input collection."
                        )

        for permutation in self.event_symmetries.permutations:
            for cycle in permutation:
                cycle_sources = {
                    tuple(self.product_particles[self.event_particles[index]].sources)
                    for index in cycle
                }

                if len(cycle_sources) > 1:
                    cycle_names = ", ".join(self.event_particles[index] for index in cycle)
                    raise ValueError(
                        f"Event particles ({cycle_names}) are related by a symmetry but their products "
                        f"are assigned to different inputs. "
                        f"Symmetric particles must share the same input collections."
                    )

    def product_source_names(self, event_particle: str) -> Tuple[Optional[str], ...]:
        """ The name of the input collection assigned to every product of an event particle. """
        return tuple(
            self.input_names[source] if source >= 0 else None
            for source in self.product_particles[event_particle].sources
        )

    def __str__(self):
        info = []

        info.append("Event Info")
        info.append("=" * 80)

        info.append("\nInputs")
        info.append("-" * 80)
        for source, features in self.input_features.items():
            info.append(f"{source}")

            for feature in features[:-1]:
                info.append(f"├───{feature.name}")
            
            info.append(f"╰───{features[-1].name}")

        info.append("\nAssignments")
        info.append("-" * 80)

        for particle, daughters in self.product_particles.items():
            info.append(f"{particle}")

            for daughter in daughters[:-1]:
                info.append(f"├───{daughter}")

            info.append(f"╰───{daughters[-1]}")

        info.append("\nSymmetry Groups")
        info.append("-" * 80)

        info.append(f"Event: {create_group_string(self.event_symbolic_group, self.event_particles)}")
        for particle, group in self.product_symbolic_groups.items():
            info.append(f"{particle}: {create_group_string(group, self.product_particles[particle])}")

        return "\n".join(info)

    def __repr__(self):
        return str(self)

    def normalized_features(self, input_name: str) -> NDArray[bool]:
        return np.array([feature.normalize for feature in self.input_features[input_name]])

    def log_features(self, input_name: str) -> NDArray[bool]:
        return np.array([feature.log_scale for feature in self.input_features[input_name]])

    @cached_property
    def event_symbolic_group(self) -> SymbolicPermutationGroup:
        return complete_symbolic_symmetry_group(*self.event_symmetries)

    @cached_property
    def event_permutation_group(self) -> PermutationGroup:
        return complete_symmetry_group(*self.event_symmetries)

    @cached_property
    def ordered_event_transpositions(self) -> Set[List[int]]:
        return set(chain.from_iterable(
            e.transpositions()
            for e in self.event_symbolic_group.elements
        ))

    @cached_property
    def event_transpositions(self) -> Set[Tuple[int, int]]:
        return set(map(tuple, map(sorted, self.ordered_event_transpositions)))

    @cached_property
    def event_equivalence_classes(self) -> Set[FrozenSet[FrozenSet[int]]]:
        num_particles = self.event_symmetries.degree
        group = self.event_symbolic_group
        sets = map(frozenset, power_set(range(num_particles)))
        return set(frozenset(frozenset(g(x) for x in s) for g in group.elements) for s in sets)

    @cached_property
    def product_permutation_groups(self) -> ODict[str, PermutationGroup]:
        output = []

        for name, (degree, symmetries) in self.product_symmetries.items():
            symmetries = [] if symmetries is None else symmetries
            permutation_group = complete_symmetry_group(degree, symmetries)
            output.append((name, permutation_group))

        return OrderedDict(output)

    @cached_property
    def product_symbolic_groups(self) -> ODict[str, SymbolicPermutationGroup]:
        output = []

        for name, (degree, symmetries) in self.product_symmetries.items():
            symmetries = [] if symmetries is None else symmetries
            permutation_group = complete_symbolic_symmetry_group(degree, symmetries)
            output.append((name, permutation_group))

        return OrderedDict(output)

    def num_features(self, input_name: str) -> int:
        return len(self.input_features[input_name])

    def input_type(self, input_name: str) -> InputType:
        return self.input_types[input_name].upper()

    @staticmethod
    def parse_list(list_string: str):
        return tuple(map(str.strip, list_string.strip("][").strip(")(").split(",")))

    @staticmethod
    def construct_mapping(variables: Iterable[str]) -> ODict[str, int]:
        return OrderedDict(map(reversed, enumerate(variables)))

    @staticmethod
    def apply_mapping(permutations: Permutations, mapping: Dict[str, int]) -> MappedPermutations:
        return [
            [
                tuple(
                    mapping[element]
                    for element in cycle
                )
                for cycle in permutation
            ]
            for permutation in permutations
        ]

    @classmethod
    def read_from_ini(cls, filename: str):
        config = ConfigParser()
        config.read(filename)

        if "INPUTS" in config:
            features_types = OrderedDict([
                (key.upper(), val)
                for key, val in config["INPUTS"].items()
            ])
        else:
            features_types = OrderedDict([("SOURCE", "sequential")])

        print(features_types)
        source_features = OrderedDict(
            (
                key,
                [
                    (
                        name,
                        "normalize" in normalize.lower() or "true" in normalize.lower(),
                        "log" in normalize.lower()
                    )
                    for name, normalize in config[key].items()
                ]
            )
            for key in features_types
        )

        event_particles = cls.parse_list(config["EVENT"]["particles"])
        event_permutations = config["EVENT"]["permutations"]

        targets = OrderedDict()
        for key in event_particles:
            target_jets = cls.parse_list(config[key]["jets"])
            target_permutations = config[key]["permutations"]
            targets[key] = (target_jets, target_permutations)

        return cls(features_types, source_features, event_particles, event_permutations, targets)

    @classmethod
    def read_from_yaml(cls, filename: str):
        with open(filename, 'r') as file:
            config = yaml_load(file, Loader)

        # Extract input feature information.
        # ----------------------------------
        input_types = OrderedDict()
        input_features = OrderedDict()

        for input_type in config[SpecialKey.Inputs]:
            current_inputs = with_default(config[SpecialKey.Inputs][input_type], default={})

            for input_name, input_information in current_inputs.items():
                input_types[input_name] = input_type.upper()
                input_features[input_name] = tuple(
                    FeatureInfo(
                        name=name,
                        normalize=("normalize" in normalize.lower()) or ("true" in normalize.lower()),
                        log_scale="log" in normalize.lower()
                    )

                    for name, normalize in input_information.items()
                )

        # Extract event and permutation information.
        # ------------------------------------------
        permutation_config = key_with_default(config, SpecialKey.Permutations, default={})

        event_names = tuple(config[SpecialKey.Event].keys())
        event_permutations = key_with_default(permutation_config, SpecialKey.Event, default=[])
        event_permutations = expand_permutations(event_permutations)
        event_particles = Particles(event_names, event_permutations)

        product_particles = OrderedDict()
        for event_particle in event_particles:
            products = config[SpecialKey.Event][event_particle]

            product_names = [
                next(iter(product.keys())) if isinstance(product, dict) else product
                for product in products
            ]

            product_sources = [
                next(iter(product.values())) if isinstance(product, dict) else None
                for product in products
            ]

            input_names = list(input_types.keys())
            for source in product_sources:
                if source is not None and source not in input_names:
                    raise ValueError(
                        f"Product of {event_particle} is assigned to the unknown input '{source}'. "
                        f"Available inputs are: {input_names}."
                    )

            product_sources = [
                input_names.index(source) if source is not None else -1
                for source in product_sources
            ]

            product_permutations = key_with_default(permutation_config, event_particle, default=[])
            product_permutations = expand_permutations(product_permutations)

            product_particles[event_particle] = Particles(product_names, product_permutations, product_sources)

        # Extract Regression Information.
        # -------------------------------
        regressions = key_with_default(config, SpecialKey.Regressions, default={})
        regressions = feynman_fill(regressions, event_particles, product_particles, constructor=list)

        # Fill in any default parameters for regressions such as gaussian type.
        regressions = feynman_map(
            lambda raw_regressions: [
                RegressionInfo(*(regression if isinstance(regression, list) else [regression]))
                for regression in raw_regressions
            ],
            regressions
        )

        # Extract Classification Information.
        # -----------------------------------
        classifications = key_with_default(config, SpecialKey.Classifications, default={})
        classifications = feynman_fill(classifications, event_particles, product_particles, constructor=list)

        return cls(
            input_types,
            input_features,
            event_particles,
            product_particles,
            regressions,
            classifications
        )
