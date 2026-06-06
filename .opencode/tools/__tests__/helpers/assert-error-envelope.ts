export const assertErrorPrefix = (value: string, expectedPrefix: string): void => {
  if (!value.startsWith(expectedPrefix)) {
    throw new Error(
      `Expected error prefix '${expectedPrefix}' but got '${value.slice(0, expectedPrefix.length + 20)}'`,
    );
  }
};

export const assertContains = (value: string, expectedSnippet: string): void => {
  if (!value.includes(expectedSnippet)) {
    throw new Error(`Expected output to contain '${expectedSnippet}' but got '${value}'`);
  }
};
