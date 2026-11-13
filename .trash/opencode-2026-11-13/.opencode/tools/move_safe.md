# move_safe

Non-destructive move wrapper.

```json
{ "source": "src/a.ts", "destination": "src/b.ts" }
```

- Rejects existing destination (`ERROR [DEST_EXISTS]`)
- Rejects out-of-repo paths
- No overwrite/trash options
