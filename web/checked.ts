export function assert(holds: boolean, what: string): void {
  if (what.length === 0) throw new Error("an assertion must say what it checks");
  if (!holds) throw new Error(`${what} does not hold`);
}

export function at<T>(a: ArrayLike<T>, i: number): T {
  if (!(i >= 0 && i < a.length)) throw new RangeError(`index ${i} is outside 0..${a.length}`);
  if (!Number.isInteger(i)) throw new RangeError(`index ${i} is not a whole number`);
  return a[i] as T;
}

export function must<T>(v: T | null | undefined, what: string): T {
  if (what.length === 0) throw new Error("a missing value must be named");
  if (v === null || v === undefined) throw new Error(`${what} is missing`);
  return v;
}
