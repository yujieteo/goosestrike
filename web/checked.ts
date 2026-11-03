export function at<T>(a: ArrayLike<T>, i: number): T {
  if (!(i >= 0 && i < a.length)) throw new RangeError(`index ${i} is outside 0..${a.length}`);
  return a[i] as T;
}

export function must<T>(v: T | null | undefined, what: string): T {
  if (v === null || v === undefined) throw new Error(`${what} is missing`);
  return v;
}
