# particula.runnable

*module*

Runnable process utilities for modifying Aerosol instances.

This module provides abstract and concrete classes for processes
that can modify or transform an Aerosol object over a simulation
time step.

**Classes**

- `- Runnable` — Abstract base class requiring 'rate' and 'execute' methods.
- `- RunnableSequence` — Manages a sequence of Runnable processes.

**Examples**

```py title="Chaining processes"
process1 = SomeRunnableProcess()
process2 = AnotherRunnableProcess()
chained = process1 | process2
final_aerosol = chained.execute(aerosol, time_step=1.0)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L1-L1)

## particula.runnable.AVAILABLE_TQDM

*attribute* · *module attribute*

```python
AVAILABLE_TQDM = True
```

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L30-L30)

## particula.runnable.RunnableABC

*class*

```python
class RunnableABC(ABC)
```

Bases: `ABC`

Abstract base class for processes modifying an Aerosol instance.

This class enforces the implementation of a process rate calculation
and an execution method that modifies the Aerosol in-place. Subclasses
must implement both 'rate' and 'execute', which define how the process
affects the Aerosol over a time step.

**Functions**

- `- rate` — Calculate and return the rate of the process.
- `- execute` — Apply the process logic to the Aerosol over a specified
time.
- `- __or__` — Chain two processes using the '|' operator.

**Examples**

```py title="Defining a Custom Process"
class CustomProcess(Runnable):
    def rate(self, aerosol):
        return 42

    def execute(self, aerosol, time_step, sub_steps=1):
        # Modify aerosol here
        return aerosol
```

> **References**
>
> - No references available yet.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L36-L129)

### particula.runnable.RunnableABC.__or__

*method*

```python
def __or__(other: RunnableABC) -> RunnableSequence
```

Chain this Runnable with another using the '|' operator.

This method enables an easy way to sequence processes, so they
can be executed in a defined order.

**Parameters**

- `other` (`RunnableABC`) — Another Runnable to append after this one.

**Returns**

- (`RunnableSequence`) — A sequence containing both processes.

**Examples**

```py title="Chaining two processes"
combined_process = process1 | process2
final_aerosol = combined_process.execute(aerosol, time_step=1.0)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L108-L129)

### particula.runnable.RunnableABC.execute

*method* · *abstract*

```python
def execute(aerosol: Aerosol, time_step: float, sub_steps: int = 1) -> Aerosol
```

Execute the process, modifying the Aerosol in-place over a time step.

**Parameters**

- `aerosol` (`Aerosol`) — The Aerosol instance to be updated.
- `time_step` (`float`) — The time step size in seconds.
- `sub_steps` (`int`) (default: `1`) — Number of sub-steps to subdivide the time step, default 1.

**Returns**

- (`Aerosol`) — - The updated Aerosol after this process runs.

**Examples**

```py title="Executing the process"
process = CustomProcess()
updated_aerosol = process.execute(my_aerosol, time_step=1.0)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L83-L106)

### particula.runnable.RunnableABC.rate

*method* · *abstract*

```python
def rate(aerosol: Aerosol) -> Any
```

Calculate and return the rate of this process for the given Aerosol.

**Parameters**

- `aerosol` (`Aerosol`) — The Aerosol instance on which to calculate the rate.

**Returns**

- (`Any`) — - Any : The computed rate of this process.

**Examples**

```py title="Using the rate method"
process = CustomProcess()
process_rate = process.rate(my_aerosol)
print(process_rate)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L65-L81)

## particula.runnable.RunnableSequence

*class*

```python
class RunnableSequence
```

A sequence of Runnable processes executed in order.

This class maintains a list of processes to be applied sequentially
to an Aerosol. Each process modifies the Aerosol and passes it along
to the next in the sequence.

**Attributes**

- `-` (`processes`) — A list of Runnable objects forming the sequence.

**Functions**

- `- add_process` — Add a Runnable to the sequence.
- `- execute` — Apply each Runnable in the sequence to an Aerosol.
- `- __or__` — Chain a new Runnable into this sequence.

**Examples**

```py title="Building and running a RunnableSequence"
sequence = RunnableSequence()
sequence.add_process(CustomProcess())
sequence.add_process(AnotherProcess())
final_aerosol = sequence.execute(aerosol, time_step=2.0)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L132-L238)

### particula.runnable.RunnableSequence.processes

*attribute* · *instance attribute*

```python
processes: list[RunnableABC] = []
```

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L161-L161)

### particula.runnable.RunnableSequence.__init__

*method*

```python
def __init__() -> None
```

Initialize the RunnableSequence.

Sets up an empty list to hold the sequence of Runnable processes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L156-L161)

### particula.runnable.RunnableSequence.__or__

*method*

```python
def __or__(process: RunnableABC) -> RunnableSequence
```

Chain another Runnable into this sequence using the '|' operator.

**Parameters**

- `process` (`RunnableABC`) — The Runnable to add.

**Returns**

- (`RunnableSequence`) — This sequence with the new Runnable appended.

**Examples**

```py
sequence = RunnableSequence()
sequence |= CustomProcess()
# or
sequence = sequence | AnotherProcess()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L220-L238)

### particula.runnable.RunnableSequence.add_process

*method*

```python
def add_process(process: RunnableABC) -> None
```

Add a Runnable to the sequence.

**Parameters**

- `process` (`RunnableABC`) — The Runnable to add.

**Examples**

```py
sequence = RunnableSequence()
sequence.add_process(CustomProcess())
```

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L163-L175)

### particula.runnable.RunnableSequence.execute

*method*

```python
def execute(aerosol: Aerosol, time_step: float, sub_steps: int = 1) -> Aerosol
```

Execute all processes in the sequence on the given Aerosol.

Each Runnable in the sequence modifies the Aerosol and passes
it to the next Runnable until all have been executed. A full cycle is
performed over each sub-step of the time step.

**Parameters**

- `aerosol` (`Aerosol`) — The Aerosol instance to be updated.
- `time_step` (`float`) — The time step size in seconds for each process.
- `sub_steps` (`int`) (default: `1`) — Number of sub-steps to subdivide the time step, default 1.

**Returns**

- (`Aerosol`) — - Aerosol : The resulting Aerosol after all processes run.

**Examples**

```py title="Executing a RunnableSequence"
sequence = RunnableSequence()
final_aerosol = sequence.execute(
    aerosol, time_step=1.0, sub_steps=4
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L177-L218)
