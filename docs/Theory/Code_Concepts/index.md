# Code Concepts

This section is the conceptual overview of the Particula codebase—what design ideas we follow, why we follow them, and where you can dig deeper.

> If something feels unclear, ask and contribute to an issue or PR—improving the docs is a meaningful contribution.

## Why read this?

- You want to _extend_ Particula without breaking existing work.  
- You need to _audit_ a calculation and trace where a number comes from.  
- You plan to _prototype_ a new physical model and wonder which files to touch.  

## Quick tour

| Topic | Start here | One–line takeaway |
|-------|------------|-------------------|
| Philosophy | [WARMED principle](Details/WARMED_principle.md) | Code must be **Writable, Agreeable, Readable, Modifiable, Executable, Debuggable**. |
| Dual Paradigm | [Design Patterns](Details/Design_Patterns.md) | Pick _functions_ for notebooks, _builders + strategies_ for experiments. |
| OO cheat‑sheet | [Object‑Oriented Patterns](Details/Object_Oriented_Patterns.md) | Strategy, Builder, Factory … explained with aerosol examples. |

## Naming rules (TL;DR)

- Functions that _return a value_ → `get_<quantity>()`  
- Classes that _encapsulate a pattern_ → `<Descriptor><PatternName>`  

Keeping to these names makes `grep`, IDE auto‑completion, and LLM help far more
effective for beginners and experts alike.


