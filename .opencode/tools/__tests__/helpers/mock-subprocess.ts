type SpawnResponse = {
  stdout?: string;
  stdoutChunks?: string[];
  stderr?: string;
  stderrChunks?: string[];
  exitCode?: number;
  timedOut?: boolean;
  hangs?: boolean;
};

type SpawnError = {
  stdout?: string;
  stderr?: string;
  message?: string;
};

type BunDollarError = {
  stdout?: string;
  stderr?: string;
  message?: string;
};

type Invocation = {
  kind: "spawnSync" | "$";
  args: string[];
  env?: Record<string, string>;
  cwd?: string;
  stdin?: string | Uint8Array;
  stdout?: "pipe";
  stderr?: "pipe";
  timeout?: number;
};

const originalBun = (globalThis as { Bun?: typeof Bun }).Bun;
let replacedBun = false;

let spawnResponse: SpawnResponse = { stdout: "", stderr: "", exitCode: 0 };
let spawnSyncResponse: SpawnResponse = { stdout: "", stderr: "", exitCode: 0 };
let spawnError: SpawnError | null = null;
let spawnSyncError: SpawnError | null = null;
let dollarText = "";
let dollarError: BunDollarError | null = null;

const invocations: Invocation[] = [];
let killCount = 0;

const ensureBun = (): typeof Bun => {
  if ((globalThis as { Bun?: typeof Bun }).Bun) {
    return (globalThis as { Bun: typeof Bun }).Bun;
  }

  const bunShim = {
    spawnSync: () => ({ stdout: Buffer.from(""), stderr: Buffer.from(""), exitCode: 0 }),
    $: () => ({ text: async () => "" }),
  } as unknown as typeof Bun;
  (globalThis as { Bun?: typeof Bun }).Bun = bunShim;
  replacedBun = true;
  return bunShim;
};

const bunRef = ensureBun();
const originalSpawnSync = bunRef.spawnSync;
const originalSpawn = bunRef.spawn;
const originalDollar = bunRef.$;

const createStreamFromChunks = (chunks: string[] | undefined): ReadableStream<Uint8Array> => {
  const encoder = new TextEncoder();
  const values = (chunks && chunks.length > 0 ? chunks : [""]).map((chunk) => encoder.encode(chunk));
  let index = 0;
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (index >= values.length) {
        controller.close();
        return;
      }
      controller.enqueue(values[index] ?? new Uint8Array());
      index += 1;
    },
  });
};

const toArgv = (input: unknown[]): string[] =>
  input.flatMap((part) => {
    if (Array.isArray(part)) {
      return part.map(String);
    }
    return [String(part)];
  });

export const installSubprocessMocks = (): void => {
  bunRef.spawnSync = ((args: string[] | {
    cmd?: string[];
    env?: Record<string, string>;
    cwd?: string;
    stdin?: string | Uint8Array;
    stdout?: "pipe";
    stderr?: "pipe";
    timeout?: number;
  }) => {
    const capturedArgs = Array.isArray(args)
      ? [...args]
      : Array.isArray(args?.cmd)
      ? [...args.cmd]
      : [String(args)];
    invocations.push({
      kind: "spawnSync",
      args: capturedArgs,
      env: Array.isArray(args) ? undefined : args?.env,
      cwd: Array.isArray(args) ? undefined : args?.cwd,
      stdin: Array.isArray(args)
        ? undefined
        : typeof args?.stdin === "string"
        ? args.stdin
        : args?.stdin
        ? Buffer.from(args.stdin).toString("utf8")
        : undefined,
      stdout: Array.isArray(args) ? undefined : args?.stdout,
      stderr: Array.isArray(args) ? undefined : args?.stderr,
      timeout: Array.isArray(args) ? undefined : args?.timeout,
    });
    if (spawnSyncError) {
      const err = new Error(spawnSyncError.message ?? "") as Error & {
        stdout?: Buffer;
        stderr?: Buffer;
      };
      err.stdout = Buffer.from(spawnSyncError.stdout ?? "");
      err.stderr = Buffer.from(spawnSyncError.stderr ?? "");
      throw err;
    }
    return {
      stdout: Buffer.from(spawnSyncResponse.stdout ?? ""),
      stderr: Buffer.from(spawnSyncResponse.stderr ?? ""),
      exitCode: spawnSyncResponse.exitCode ?? 0,
      timedOut: spawnSyncResponse.timedOut ?? false,
    };
  }) as typeof bunRef.spawnSync;

  bunRef.spawn = ((args: string[] | { cmd?: string[]; env?: Record<string, string> }) => {
    const capturedArgs = Array.isArray(args)
      ? [...args]
      : Array.isArray(args?.cmd)
      ? [...args.cmd]
      : [String(args)];
    invocations.push({ kind: "spawnSync", args: capturedArgs, env: Array.isArray(args) ? undefined : args?.env });
    if (spawnError) {
      throw new Error(spawnError.message ?? "");
    }
    let killed = false;
    let resolveExit: ((exitCode: number) => void) | undefined;
    const exited = spawnResponse.hangs
      ? new Promise<number>((resolve) => {
          resolveExit = resolve;
        })
      : Promise.resolve(spawnResponse.exitCode ?? 0);
    return {
      stdout: createStreamFromChunks(spawnResponse.stdoutChunks ?? [spawnResponse.stdout ?? ""]),
      stderr: createStreamFromChunks(spawnResponse.stderrChunks ?? [spawnResponse.stderr ?? ""]),
      kill() {
        if (killed) return;
        killed = true;
        killCount += 1;
        resolveExit?.(0);
      },
      get exited() {
        return exited;
      },
    } as unknown as ReturnType<typeof bunRef.spawn>;
  }) as typeof bunRef.spawn;

  bunRef.$ = ((strings: TemplateStringsArray, ...values: unknown[]) => {
    const segments: unknown[] = [];
    for (let i = 0; i < strings.length; i += 1) {
      const literal = strings[i];
      if (literal.trim()) {
        segments.push(literal.trim());
      }
      if (i < values.length) {
        segments.push(values[i]);
      }
    }
    const invocation: Invocation = { kind: "$", args: toArgv(segments) };
    invocations.push(invocation);

    const text = async () => {
      if (dollarError) {
        const err = new Error(dollarError.message ?? "mock subprocess failure") as Error & {
          stdout?: Buffer;
          stderr?: Buffer;
        };
        err.stdout = Buffer.from(dollarError.stdout ?? "");
        err.stderr = Buffer.from(dollarError.stderr ?? "");
        return Promise.reject(err);
      }

      return dollarText;
    };

    const command = {
      cwd(value: string) {
        invocation.cwd = value;
        return command;
      },
      text,
    };

    return command;
  }) as typeof bunRef.$;
};

export const setSpawnResponse = (response: SpawnResponse): void => {
  spawnError = null;
  spawnSyncError = null;
  spawnResponse = { stdout: "", stderr: "", exitCode: 0, ...response };
  spawnSyncResponse = { stdout: "", stderr: "", exitCode: 0, ...response };
};

export const setSpawnError = (error: SpawnError): void => {
  spawnError = error;
  spawnSyncError = error;
};

export const setDollarText = (value: string): void => {
  dollarError = null;
  dollarText = value;
  // AST-grep wrappers use Bun.spawn so process lifecycle can be bounded and
  // reaped; keep the legacy helper convenient for their existing fixtures.
  spawnError = null;
  spawnResponse = { stdout: value, stderr: "", exitCode: 0 };
};

export const setDollarError = (error: BunDollarError): void => {
  dollarError = error;
  spawnError = null;
  spawnResponse = {
    stdout: error.stdout ?? "",
    stderr: error.stderr ?? "",
    exitCode: 2,
  };
};

export const getInvocations = (): Invocation[] => [...invocations];

export const getKillCount = (): number => killCount;

/** Reset mock behavior/captured calls while keeping monkeypatches installed. */
export const resetSubprocessMocks = (): void => {
  spawnResponse = { stdout: "", stderr: "", exitCode: 0 };
  spawnSyncResponse = { stdout: "", stderr: "", exitCode: 0 };
  spawnError = null;
  spawnSyncError = null;
  dollarText = "";
  dollarError = null;
  invocations.length = 0;
  killCount = 0;
};

/** Restore Bun globals to original state; call from afterEach for isolation. */
export const restoreSubprocessMocks = (): void => {
  bunRef.spawnSync = originalSpawnSync;
  bunRef.spawn = originalSpawn;
  bunRef.$ = originalDollar;
  resetSubprocessMocks();

  if (replacedBun) {
    if (originalBun) {
      (globalThis as { Bun?: typeof Bun }).Bun = originalBun;
    } else {
      delete (globalThis as { Bun?: typeof Bun }).Bun;
    }
    replacedBun = false;
  }
};
