// Shared precondition check for the figure sources: it throws when the condition is false.
export function assert(holds, what) {
  if (typeof what !== 'string' || what.length === 0) throw new Error('an assertion must say what it checks');
  if (!holds) throw new Error(`${what} does not hold`);
}
