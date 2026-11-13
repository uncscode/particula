# move_overwrite

Explicit overwrite move wrapper.

```json
{ "source": "src/new.ts", "destination": "src/current.ts" }
```

- Allows replacing existing destination
- Uses rollback attempt with `.move-backup` if move fails
- Rejects out-of-repo paths
