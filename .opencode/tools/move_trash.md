# move_trash

Soft-delete wrapper. Moves only to `.trash/<relative-path>`.

```json
{ "source": "src/deprecated.ts" }
```

- No destination parameter
- Rejects already-in-trash and trash conflicts
- Rejects out-of-repo paths
