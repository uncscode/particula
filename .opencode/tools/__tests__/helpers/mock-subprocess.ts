type SpawnResponse = {
  stdout?: string;
  stderr?: string;
  exitCode?: number;
  timedOut?: boolean;
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
};

const originalBun = (globalThis as { Bun?: typeof Bun }).Bun;
let replacedBun = false;

let spawnResponse: SpawnResponse = { stdout: "", stderr: "", exitCode: 0 };
let spawnError: SpawnError | null = null;
let dollarText = "";
let dollarError: BunDollarError | null = null;

const invocations: Invocation[] = [];

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
const originalDollar = bunRef.$;

const toArgv = (input: unknown[]): string[] =>
  input.flatMap((part) => {
    if (Array.isArray(part)) {
      return part.map(String);
    }
    return [String(part)];
  });

export const installSubprocessMocks = (): void => {
  bunRef.spawnSync = ((args: string[] | { cmd?: string[] }) => {
    const capturedArgs = Array.isArray(args)
      ? [...args]
      : Array.isArray(args?.cmd)
      ? [...args.cmd]
      : [String(args)];
    invocations.push({ kind: "spawnSync", args: capturedArgs });
    if (spawnError) {
      const err = new Error(spawnError.message ?? "") as Error & {
        stdout?: Buffer;
        stderr?: Buffer;
      };
      err.stdout = Buffer.from(spawnError.stdout ?? "");
      err.stderr = Buffer.from(spawnError.stderr ?? "");
      throw err;
    }
    return {
      stdout: Buffer.from(spawnResponse.stdout ?? ""),
      stderr: Buffer.from(spawnResponse.stderr ?? ""),
      exitCode: spawnResponse.exitCode ?? 0,
      timedOut: spawnResponse.timedOut ?? false,
    };
  }) as typeof bunRef.spawnSync;

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
    invocations.push({ kind: "$", args: toArgv(segments) });

    if (dollarError) {
      const err = new Error(dollarError.message ?? "mock subprocess failure") as Error & {
        stdout?: Buffer;
        stderr?: Buffer;
      };
      err.stdout = Buffer.from(dollarError.stdout ?? "");
      err.stderr = Buffer.from(dollarError.stderr ?? "");
      return { text: async () => Promise.reject(err) };
    }

    return { text: async () => dollarText };
  }) as typeof bunRef.$;
};

export const setSpawnResponse = (response: SpawnResponse): void => {
  spawnError = null;
  spawnResponse = { stdout: "", stderr: "", exitCode: 0, ...response };
};

export const setSpawnError = (error: SpawnError): void => {
  spawnError = error;
};

export const setDollarText = (value: string): void => {
  dollarError = null;
  dollarText = value;
};

export const setDollarError = (error: BunDollarError): void => {
  dollarError = error;
};

export const getInvocations = (): Invocation[] => [...invocations];

/** Reset mock behavior/captured calls while keeping monkeypatches installed. */
export const resetSubprocessMocks = (): void => {
  spawnResponse = { stdout: "", stderr: "", exitCode: 0 };
  spawnError = null;
  dollarText = "";
  dollarError = null;
  invocations.length = 0;
};

/** Restore Bun globals to original state; call from afterEach for isolation. */
export const restoreSubprocessMocks = (): void => {
  bunRef.spawnSync = originalSpawnSync;
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
