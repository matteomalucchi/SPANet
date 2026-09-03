# Event Specification Format

The first step to training SPANets is to define the topology of your target event. To do this, `SPANet`uses a definition `.yaml` file which contains the features and jet information for your event. This will describe both the inputs and outputs for your model, along with any related symmetries.

The structure of the `.yaml` file will follows a standard format. Special keys which must be exactly as shown will be in `CAPITALCASE`. Custom keys which may modified for your event will be in `lower_case_with_underscores`
```yaml
INPUTS:
    SEQUENTIAL:
        sequential_input_1:
            feature_1: feature_option_1
            feature_2: feature_option_2
            feature_3: feature_option_3
            ...
            
        sequential_input_2:
            feature_1: feature_option_1
            feature_2: feature_option_2
            feature_3: feature_option_3
            ...
        ...
        
    GLOBAL:
        global_input_1:
            feature_1: feature_option_1
            feature_2: feature_option_2
            feature_3: feature_option_3
            ...
        ...
        
EVENT:
    event_particle_1:
        - decay_product_1: sequential_input_1
        - decay_product_2: sequential_input_1
        ...
        
    event_particle_2:
        - decay_product_1: sequential_input_1
        - decay_product_2: sequential_input_1
        ...
    
    ...

PERMUTATIONS:
    EVENT:
        - [ event_particle, event_particle ]
        ...
        
    event_particle_1:
        - [ decay_product, decay_product ]
        ...
        
    event_particle_2:
        - [ decay_product, decay_product ]
        ...
    ...
   
REGRESSIONS:
    ... (WIP Explained Later)

CLASSIFICATIONS:
    ... (WIP Explained Later)
```

We will now go over each of the sections to explain what must be included. 
You may also view an example of a complete event file
in [`ttbar.yaml`](../event_files/full_hadronic_ttbar.yaml).

## `INPUTS`
The first **reuquired** section. Inputs will contain a description of the features that will be fed in as input to SPANet. Each input should have a unique name. They will later be used in how you define your dataset. The exact names are not important
as long as they are unique.

There are two types of inputs:
- **SEQUENTIAL** inputs represent variable length inputs for each event.
  These may include objects such as hadronic jets, leptons, neutrinos, etc. We require that the event contains at least one sequential input. These inputs will also define the potential reconstruction targets.
- **GLOBAL** inputs represent features which exist for the entire event.
  There exists only a single instance of the inputs for every event.
  Examples include neutrino missing energy.

Each input contains one or more *features*. These are the observable values associated with each input. Each feature is also given a unique name which will be used when creating the dataset. Each feature can have association several options which define how `SPANet` will pre-process the feature. Valid options include:

| FEATURE_OPTION     | Description |
| :--------------:   | ----------- |
| `none`             | No pre-processing applied            |
| `log`              | Scale the feature on a `log` scale | 
| `normalize`        | Normalize the feature based on training dataset statistics |
| `log_normalize`    | First apply a `log` scale and then normalize |

## `EVENT`
The second **required** section. This will contain a simplified Feynman diagram of your event. We require that events processed by SPANet follow a particular two-level structure. We split the event into
1. **Event Particles:** The first level of the Feynmen Diagram. These are typically non-observable particles which we are interested in studying. These particles are required to decay into other particles.
2. **Decay Products:** The second level will contain observable decay products. These are required to be particles which will have reconstruction targets associated with them. Decay products may correspond to a specific sequential input. You can define the correspondence by adding the name of the sequential input after the decay product `decay_product: sequential_input`.

We describe this Feynman diagram structure with a simple two layer tree. Give each event particle a unique name. Decay particles may repeat names as long as they belong to different event particles.

### Exclusive input collections

By default every reconstructable input is concatenated into a **single merged collection** which any decay
product may be assigned to. Naming an input after a decay product only tells SPANet how to interpret the
indices stored in your dataset, it does not restrict what the network may assign. In the example below the
network is free to assign `q1` to a `JetHiggs` vector.

The `assignment_source_exclusivity` option (see [`Options.md`](Options.md)) makes those assignments
**exclusive** instead. With

```yaml
INPUTS:
  SEQUENTIAL:
    JetHiggs:
      ...
    JetVBF:
      ...

EVENT:
  h1:
    - b1: JetHiggs
    - b2: JetHiggs
  h2:
    - b3: JetHiggs
    - b4: JetHiggs
  vbf:
    - q1: JetVBF
    - q2: JetVBF
```

and `"assignment_source_exclusivity": true`, `q1` and `q2` can only ever be assigned to a `JetVBF` vector
and `b1`-`b4` can only ever be assigned to a `JetHiggs` vector, both during training and when predicting.
The constraint is implemented by masking out the forbidden vectors in the assignment distribution of each
decay product, so the distribution is normalized only over the legal combinations and the training loss,
the validation metrics and the predictions are all constrained in the same way.

Decay products which do not name an input remain free to select a vector from any sequential input, even
when the option is enabled.

Because two decay products related by a symmetry must be indistinguishable, they are required to come from
the same input. The same holds for two event particles related by an event-level symmetry. SPANet warns
about an event file which violates this when reading it, and refuses to build a network from it when
`assignment_source_exclusivity` is enabled.

### Migrating an existing configuration

Existing event files and options files keep their previous behaviour with no changes: the option defaults
to `false` and the outputs of `predict.py` are unchanged. To adopt exclusive collections:

1. **Event file** — give every decay product the input it belongs to,
   `decay_product: input_name`. Products of particles related by a symmetry must
   name the same input. Nothing else in the event file changes.
2. **Options file** — add `"assignment_source_exclusivity": true`.
3. **Retrain.** The option changes the shape of the assignment distribution, so an existing checkpoint
   trained without it has learnt to spread probability over the vectors which are now masked out. The
   checkpoint still loads, since no weights are added or removed, but retraining is what actually buys you
   the improved pairing.
4. **Predict with `--local_indices`** if you want the predicted indices written in the index-space of the
   collection each product belongs to, matching the convention of the indices in your input dataset. See
   below.

### Indices in the prediction output

Internally SPANet concatenates every reconstructable input into a single sequence, so a target of the
second input is offset by the size of the first. For the example above, `JetHiggs` occupies indices
`0..N_higgs-1` and `JetVBF` occupies `N_higgs..N_higgs+N_vbf-1`.

`predict.py` writes the assignments in that merged index-space by default. Pass `--local_indices` to
instead write each decay product as an index into its own collection, which is the same convention used by
the `TARGETS` indices of the input dataset. Each output dataset of a product which names an input also
carries two HDF5 attributes, `input` (the name of the collection) and `merged_indices` (whether the
indices are in the merged space), so the output is unambiguous either way.

Without `assignment_source_exclusivity`, the network can predict a vector outside of the collection a
product is assigned to. `--local_indices` detects this, keeps the merged indices for that product and
prints a warning rather than emitting an index into the wrong collection.



## `PERMUTATIONS`
Describe the symmetries allowed in during assignment. You may specify an event-level symmetry group over event particles with the special keyword.
```yaml
EVENT:
    - [ event_particle, event_particle ]
    ...
```
Decay product symmetry groups may be specicied with their associated event particle name.
```yaml
event_particle_1:
    - [ decay_product, decay_product ]
    ...
```

SPANet supports describing permutation groups as products of complete symmetry goups G = S_1 x S_2 x ... or using an explicit collection of generating cycles.

### Complete Symmetric Groups
In order to define symmetric permutation groups, you simply describe which particles or jets belong to each of the fully symmetric groups. This is expressed as a list of lists each of which contain the names of the connected particles or jets

For example
```yaml
EVENT:
    - [ event_particle_1, event_particle_2 ]
    - [ event_particle_3, event_particle_4 ]
```
will define an event symmetric group where the first two particles are interchangeable, and the last two particles are interchangeable.

Any groupings of three or more particles will mean that **ALL** of the particles are symmetric with each other. Any elements not present in the permutations description will be assumed to be invariant with only itself.

For example, using the same four particles as above ` [event_particle_1, event_particle_2, event_particle_3] ` defines a group with the first three particles are completely invariant with respect to each other but the final particle `event_particle_4` is invariant with nothing.

### Explicit Permutation groups
You can also define custom permutation groups using explicit cycles. These cycles will be used as the generators for a general permutation group. Cycles are defined using nested lists instead of the simple lists above. Each list of lists defines a disjoint cycle so `[[1, 2, 3], [4, 5]] = (1,2,3)(4,5)`. See the following article for more information on the disjoint cycle notation for permutations. [https://groupprops.subwiki.org/wiki/Cycle_decomposition_for_permutations](https://groupprops.subwiki.org/wiki/Cycle_decomposition_for_permutations)

For example
```yaml
EVENT:
    - [ [p1, p2], [p3, p4] ]
```
Will define an event permutation group where `p1` may only be swapped with `p2` if `p3` is simultaneously swapped with `p4`. This defines a different group than 
```yaml
EVENT:
    - [[p1, p2]]
    - [[p3, p4]]
```
Where the two pairs may be swapped independently of each other. 

## `REGRESSIONS` & `CLASSIFICATIONS`
These sections define the additional regression and classification targets associated to any part of the event tree. These are optional and may be left out if you do not wish to train a model to perform these tasks. These two sections have identical structure. We will use `REGRESSIONS` for these descriptions. 

```yaml
REGRESSIONS:
    EVENT: 
        - event_regression_1
        - event_regression_2
    event_particle_1:
        - PARTICLE:
            - event_particle_1_regression_1
            - event_particle_1_regression_2
        - decay_product_1:
            - decay_product_1_regression_1
            - decay_product_1_regression_2
        - decay_product_2:
            - decay_product_2_regression_1
            - decay_product_2_regression_2
        ...
    ...
```

You may add additional regression targets to any point in the Feynman diagram. The `EVENT` targets will be defined for the entire event and can include things such as system invariant mass. You can also add regression targets to individual event particles with the special `PARTICLE` tag. Finally, you can add regression targets to each individual decay products within each event particle. Any symmetric event particles or decay products must define the same regression targets.

**This feature is currently considered a work in progress and may not be fully compatible with the symmetric structures defined above. Only the `EVENT` regression and classifications are currently fully supported.**

## Example
Refer to the Event File section of the [`ttbar` Example Guide](TTBar.md) for a description of the `ttbar.yaml` example event file.